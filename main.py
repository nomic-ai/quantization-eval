from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import uuid

from util import top_tokens_by_layer, top_tokens, plot_attention_heads_by_layer

print("===============loading model")
precision = "bf16"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"
model = model.to(device)

def main(prompt, max_new_tokens=50, images=False):

    print("===============setting hook")

    # clear pre-existing hooks
    for _, module in model.named_modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()

    activation_sequences = []

    def collect_activations():
        def hook(model, input, output):
            # Capture full sequence representations
            activation_sequences.append(output.detach().cpu().numpy())
        return hook

    # Register hook on the final norm layer before lm_head
    hook = model.model.norm.register_forward_hook(collect_activations())

    print("===============setting up dirs")
    output_dir = os.path.join("model_outputs", precision, str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logits"), exist_ok=True)

    print("===============setting up prompt")

    # format prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )

    # tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(device)

    # Store generated tokens and their logits at each intermediate generation step
    generated_tokens = []
    current_input_ids = input_ids
    generated_token_ids = []

    print("===============starting generation")

    for step in tqdm(range(max_new_tokens)):
        with torch.no_grad():
            outputs = model(
                current_input_ids,
                output_attentions=True,
                return_dict=True
            )
        
        # Get logits for the next token
        token_logits = outputs.logits[0]
        next_token_logits = token_logits[-1]
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_token_ids.append(next_token.item())
        generated_tokens.append(next_token)

        # Save logit distribution for this step
        token_data = []
        for token_id in range(len(next_token_logits)):
            token = tokenizer.decode([token_id])
            logit = next_token_logits[token_id].item()
            token_data.append((token_id, token, logit))

        df = pd.DataFrame(token_data, columns=['Token ID', 'Token', 'Logit'])
        df = df.sort_values(by='Logit', ascending=False)[:10]
        df.to_csv(
            os.path.join(output_dir, f"logits/step_{step}_top_10_logit_distribution.csv"), 
            escapechar='\\',
            index=False
        )

        # Sample next token
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token)

        # Update input_ids for next iteration
        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

        # Optional: break if we generate an end token
        if next_token.item() == tokenizer.eos_token_id:
            break

    hook.remove()

    # save embeddings
    embeddings_data = {
        'embeddings': activation_sequences,
        'prompt_tokens': input_ids[0].cpu().numpy().tolist(),
        'generated_tokens': generated_token_ids,
        'token_strings': [tokenizer.decode([tid]) for tid in generated_token_ids]
    }
    np.save(os.path.join(output_dir, 'embedding_sequences.npy'), embeddings_data, allow_pickle=True)


    # Save the full generation with all its special tokens
    generated_text = tokenizer.decode(current_input_ids[0])
    with open(os.path.join(output_dir, 'generated_text.txt'), 'w') as f:
        f.write(generated_text)

    # Save the attention tensor with shape 
    # (n_layers, n_batch, n_heads, n_prompt_tokens, n_prompt_tokens)
    all_layers_attention = [layer.cpu().numpy() for layer in outputs.attentions]
    attention_array = np.stack(all_layers_attention)

    np.save(
        os.path.join(output_dir, 'attention_scores.npy'), 
        attention_array
    )

    # save images of the attention head matrices for each layer
    if images:
        os.makedirs(os.path.join(output_dir, "attention-by-head-by-layer"), exist_ok=True)
        for layer_num in range(16):
            plot_attention_heads_by_layer(
                layer_num,
                os.path.join(output_dir, "attention-by-head-by-layer/attention-matrices"),
                attention_array
            )

    df, fig = top_tokens_by_layer(outputs, input_ids, tokenizer)
    plt.savefig(os.path.join(output_dir, 'top_tokens_by_layer.png'))
    df.to_csv(os.path.join(output_dir, 'top_tokens_by_layer.csv'))

    df, fig = top_tokens(outputs, input_ids, tokenizer)
    plt.savefig(os.path.join(output_dir, 'top_tokens.png'))
    df.to_csv(os.path.join(output_dir, 'top_tokens.csv'))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--prompts_file', default='prompts.txt')
    args = parser.parse_args()
    with open(args.prompts_file) as f:
        prompts = f.readlines()
    for p in prompts:
        main(p)