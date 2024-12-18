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
size = "1B"
model_name = f"meta-llama/Llama-3.2-{size}-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "mps"
model = model.to(device)

def main(prompt, max_new_tokens=250, images=False, save_embeddings=False, save_attentions=False):

    if save_embeddings:
        print("===============setting hook")

        # clear pre-existing hooks
        for _, module in model.named_modules():
            module._forward_hooks.clear()
            module._backward_hooks.clear()

        activation_sequences = []

        def collect_activations():
            def hook(model, input, output):
                # Only capture the last token's embedding (shape: [1, 2048])
                activation_sequences.append(output[0, -1].detach().cpu().numpy())
            return hook

        # Register hook on the final norm layer before lm_head
        hook = model.model.norm.register_forward_hook(collect_activations())

    print("===============setting up dirs")
    output_dir = os.path.join("model_outputs", size, precision, prompt)
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
        print("run model one step...")
        with torch.no_grad():
            outputs = model(
                current_input_ids,
                output_attentions=True,
                return_dict=True
            )
        print("save top logits...")
        # Get logits for the next token and find top token
        next_token_logits = outputs.logits[0, -1]
        next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
        generated_token_ids.append(next_token.item())
        generated_tokens.append(next_token)

        # Get top 10 tokens and their logits efficiently
        top_logits, top_indices = next_token_logits.topk(10)
        
        # Batch decode tokens
        decoded_top_tokens = tokenizer.batch_decode([[i] for i in top_indices])
        
        # Create DataFrame directly from dict
        df = pd.DataFrame({
            'Token ID': top_indices.cpu().numpy(),
            'Token': decoded_top_tokens,
            'Logit': top_logits.cpu().numpy()
        })

        # Save to CSV
        df.to_csv(
            os.path.join(output_dir, f"logits/step_{step}_top_10_logit_distribution.csv"),
            escapechar='\\',
            index=False
        )

        print("save next")

        # Sample next token
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token)

        # Update input_ids for next iteration
        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

        # Optional: break if we generate an end token
        if next_token.item() == tokenizer.eos_token_id:
            break

    if save_embeddings:
        hook.remove()

        # save embeddings
        embeddings_data = {
            'embeddings': np.vstack(activation_sequences),  # Stack all embeddings into shape (n_tokens, 2048)
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
    if save_attentions:
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
        prompts = [x.replace('\n', '') for x in f.readlines()]
    for p in prompts[:1]:
        print(p)
        main(p)