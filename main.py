from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


from util import (
    create_output_folder,
    top_tokens_by_layer,
    top_tokens,
    plot_attention_heads_by_layer,
)


def do_run(
    run_id: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device: str = "cuda",
    save_attentions=False,
    images=False
):

    # load model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_new_tokens = 50

    # load prompts from file
    with open("prompts.txt") as f:
        all_prompts = [x.replace("\n", "") for x in f.readlines()]

    # create output generation metadata for each prompt
    for prompt in all_prompts:
            
        output_dir = create_output_folder(run_id, prompt)
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

        for step in tqdm(range(max_new_tokens)):
            with torch.no_grad():
                outputs = model(current_input_ids, output_attentions=True, return_dict=True)

            # Save logit distribution for this step and get next token efficiently
            next_token_logits = outputs.logits[0, -1]
            
            # Get top 10 tokens and their logits in one operation
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

            # Sample next token (using the highest logit)
            next_token = top_indices[0].unsqueeze(0).unsqueeze(0)
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

        if save_attentions:
            # Save the attention tensor with shape
            # (n_layers, n_batch, n_heads, n_prompt_tokens, n_prompt_tokens)
            all_layers_attention = [layer.cpu().numpy() for layer in outputs.attentions]
            attention_array = np.stack(all_layers_attention)
            np.save(os.path.join(output_dir, "attention_scores.npy"), attention_array)

        # save images of the attention head matrices for each layer
        if images:
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
    run_id: str,
    run_full: bool = True,
    run_4bit: bool = True,
    run_8bit: bool = True,
    run_4bitnf4: bool = True,
    device: str = "cuda",
):
    if run_full:
        do_run(f"{run_id}-full", quantization_config=None, device=device)
    if run_4bit:
        do_run(
            f"{run_id}-bnb4bit",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device=device,
        )
    if run_8bit:
        do_run(
            f"{run_id}-bnb8bit",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device=device,
        )
    if run_4bitnf4:
        do_run(
            f"{run_id}-bnb4bit-nf4",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            device=device,
        )


if __name__ == "__main__":
    typer.run(main)
