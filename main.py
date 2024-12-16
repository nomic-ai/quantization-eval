from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


from util import (
    create_output_folder,
    top_tokens_by_layer,
    top_tokens,
    plot_attention_heads_by_layer,
)


def do_run(folder_id: str, quantization_config: Optional[BitsAndBytesConfig] = None, device: str='cuda'):

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_new_tokens = 10
    prompt = "Is today before or after 2025?"

    output_dir = create_output_folder(folder_id)
    os.makedirs(os.path.join(output_dir, "logits"), exist_ok=True)

    # clear pre-existing hooks
    for _, module in model.named_modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()

    # format prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)

    # Store generated tokens and their logits at each intermediate generation step
    generated_tokens = []
    current_input_ids = input_ids

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(current_input_ids, output_attentions=True, return_dict=True)

        # Get logits for the next token
        token_logits = outputs.logits[0]
        next_token_logits = token_logits[-1]

        # Save logit distribution for this step
        token_data = []
        for token_id in range(len(next_token_logits)):
            token = tokenizer.decode([token_id])
            logit = next_token_logits[token_id].item()
            token_data.append((token_id, token, logit))

        df = pd.DataFrame(token_data, columns=["Token ID", "Token", "Logit"])
        df = df.sort_values(by="Logit", ascending=False)[:10]
        df.to_csv(
            os.path.join(
                output_dir, f"logits/step_{step}_top_10_logit_distribution.csv"
            ),
            escapechar="\\",
            index=False,
        )

        # Sample next token
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token)

        # Update input_ids for next iteration
        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

        # Optional: break if we generate an end token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Save the full generation with all its special tokens
    generated_text = tokenizer.decode(current_input_ids[0])
    with open(os.path.join(output_dir, "generated_text.txt"), "w") as f:
        f.write(generated_text)

    # Save the attention tensor with shape
    # (n_layers, n_batch, n_heads, n_prompt_tokens, n_prompt_tokens)
    all_layers_attention = [layer.cpu().numpy() for layer in outputs.attentions]
    attention_array = np.stack(all_layers_attention)

    np.save(os.path.join(output_dir, "attention_scores.npy"), attention_array)

    # save images of the attention head matrices for each layer
    os.makedirs(os.path.join(output_dir, "attention-by-head-by-layer"), exist_ok=True)
    for layer_num in range(16):
        plot_attention_heads_by_layer(
            layer_num,
            os.path.join(output_dir, "attention-by-head-by-layer/attention-matrices"),
            attention_array,
        )

    df, fig = top_tokens_by_layer(outputs, input_ids, tokenizer)
    plt.savefig(os.path.join(output_dir, "top_tokens_by_layer.png"))
    df.to_csv(os.path.join(output_dir, "top_tokens_by_layer.csv"))

    df, fig = top_tokens(outputs, input_ids, tokenizer)
    plt.savefig(os.path.join(output_dir, "top_tokens.png"))
    df.to_csv(os.path.join(output_dir, "top_tokens.csv"))


def main(
    folder_id: str, run_full: bool = True, run_4bit: bool = True, run_8bit: bool = True, run_4bitdouble:bool = True, device: str='cuda'
):
    if run_full:
        do_run(f"{folder_id}-full", quantization_config=None, device=device)
    if run_4bit:
        do_run(
            f"{folder_id}-bnb4bit",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device=device
        )
    if run_8bit:
        do_run(
            f"{folder_id}-bnb8bit",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device=device
        )
    if run_4bitdouble:
        do_run(
            f"{folder_id}-bnb4bit-double",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True),
            device=device
        )


if __name__ == "__main__":
    typer.run(main)
