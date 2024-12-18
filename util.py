# some util functions used from
# https://github.com/Pleias/Quest-Best-Tokens/blob/main/2.%20Attention%20paths%20and%20dynamics.ipynb

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
import uuid

def plot_attention_heads_by_layer(
    layer_idx, 
    base_filename, 
    attention_scores, 
    num_heads=32
):
    """
    Plot all attention matrices for a given layer

    plots 4 matrices per row
    """
    n_rows = num_heads // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, 32))

    # Add overall title with padding
    fig.suptitle(f'Attention Matrices for Layer {layer_idx}', fontsize=24, y=0.95)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot attention matrix for each head
    for head_idx in range(num_heads):
        attention_matrix = attention_scores[layer_idx, 0, head_idx]
        
        ax = axes_flat[head_idx]
        # Use faster rendering with lower resolution
        im = ax.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
        
        # Reduce text rendering overhead
        ax.set_title(f'Head {head_idx}', fontsize=10)
        
        # Remove ticks to reduce rendering overhead
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Only add labels where needed
        if head_idx >= num_heads - 4:
            ax.set_xlabel('Token Position (Target)')
        if head_idx % 4 == 0:
            ax.set_ylabel('Token Position (Source)')

    # Adjust layout to prevent overlap, with top padding for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure with layer number in filename
    filename = f"{base_filename}-layer-{layer_idx}.png"
    plt.savefig(filename)
    plt.close()
    

def clean_token(token):
    """Clean special characters from tokens."""
    # Remove the Ġ character (space indicator) and newline characters
    cleaned = token.replace('Ġ', '')
    # Remove 'Ċ' (newline character)
    cleaned = cleaned.replace('Ċ', '')
    return cleaned  

def filter_tokens(tokens, attention_weights, keep_ind=False):
    """Filter out uninteresting tokens and their attention weights."""
    TOKENS_TO_FILTER = {
        '<|begin_of_text|>', '<', '>', '=', 'ref',
        '<|source_id_start|>', '<|source_id_end|>',
        '<|source_start|>', '<|source_end|>', "Ġname"
    }

    keep_indices = [
        i for i, token in enumerate(tokens)
        if token not in TOKENS_TO_FILTER
        and not any(c in '[]<>|=' for c in token)
        and not all(c in string.punctuation for c in token)
        and token.strip(string.punctuation)
        and 'Ċ' not in token  # Filter out newline characters
    ]
    if keep_ind:
        return (
            [clean_token(tokens[i]) for i in keep_indices],
            attention_weights[..., keep_indices] if attention_weights.ndim > 1 else attention_weights[keep_indices],
            keep_indices
        )
    return (
        [clean_token(tokens[i]) for i in keep_indices],
        attention_weights[..., keep_indices] if attention_weights.ndim > 1 else attention_weights[keep_indices],
    )

def process_attention_weights(attention_weights, input_ids, tokenizer, top_k=10):
    """Process attention weights and prepare data for visualization."""
    # Convert all layers to numpy and stack
    all_layers_attention = [layer.cpu().numpy() for layer in attention_weights]
    stacked_attention = np.stack(all_layers_attention)

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    # Prepare data for plotting
    plot_data = []

    # Process each layer separately
    for layer_idx in range(stacked_attention.shape[0]):
        # Get last token attention for current layer
        layer_attention = stacked_attention[layer_idx, 0, :, -1, :]  # [heads, seq_len]
        mean_attention = layer_attention.mean(axis=0)  # Average across heads

        # Filter tokens and attention weights
        filtered_tokens, filtered_attention = filter_tokens(tokens, mean_attention)

        # Create pairs and sort
        token_attention_pairs = list(zip(filtered_tokens, filtered_attention))
        sorted_pairs = sorted(token_attention_pairs, key=lambda x: x[1], reverse=True)

        # Take top k tokens
        for token, attention in sorted_pairs[:top_k]:
            plot_data.append({
                'Layer': f'Layer {layer_idx}',
                'Token': token,
                'Attention': attention
            })

    return pd.DataFrame(plot_data)

def plot_attention_by_layer(df, num_layers, top_k=10):
    """Create faceted bar plot of attention weights with 6 subplots per row."""
    # Calculate number of rows needed with 6 plots per row
    num_rows = (num_layers + 5) // 6  # Round up division

    # Calculate figure size based on number of rows
    fig_height = 4 * num_rows  # Adjust multiplier as needed
    fig_width = 24  # Width to accommodate 6 subplots per row

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Generate a color palette with different colors for each layer
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))

    # Create subplot for each layer
    for layer_idx in range(num_layers):
        layer_name = f'Layer {layer_idx}'
        layer_data = df[df['Layer'] == layer_name]

        plt.subplot(num_rows, 6, layer_idx + 1)

        # Create horizontal bar plot with reversed order (highest attention first)
        sns.barplot(
            data=layer_data,
            y='Token',
            x='Attention',
            order=layer_data.sort_values('Attention', ascending=False)['Token'],
            color=colors[layer_idx]
        )

        plt.title(f'Layer {layer_idx}')
        plt.xlabel('Attention Weight')
        plt.ylabel('')

    plt.tight_layout()
    return fig

def top_tokens_by_layer(model_outputs, input_ids, tokenizer, top_k=10):
    """Main function to analyze attention weights and create visualization."""
    attention_weights = model_outputs.attentions

    if attention_weights:
        # Process attention weights
        df = process_attention_weights(attention_weights, input_ids, tokenizer, top_k=top_k)

        # Create visualization
        num_layers = len(attention_weights)
        fig = plot_attention_by_layer(df, num_layers, top_k=top_k)

        return df, fig
    else:
        print("Unable to get attention weights")
        return None, None
    

def process_unique_tokens(filtered_tokens, filtered_attention, keep_indices):
    """Process tokens to create unique token-position pairs with their attention weights."""
    token_info = []
    token_counts = {}

    for idx, (token, attention) in enumerate(zip(filtered_tokens, filtered_attention)):
        original_pos = keep_indices[idx]
        if token in token_counts:
            token_counts[token] += 1
            display_token = f"{token} (pos {original_pos}, occ {token_counts[token]})"
        else:
            token_counts[token] = 1
            display_token = f"{token} (pos {original_pos})"

        token_info.append((display_token, attention, original_pos))

    return token_info

def plot_mean_attention(token_info, top_k=30):
    """Create a horizontal bar plot of mean attention weights for unique token-position pairs."""
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Token': [t[0] for t in token_info],
        'Attention': [t[1] for t in token_info]
    })

    # Sort by attention weight and get top k
    df_sorted = df.nlargest(top_k, 'Attention')

    # Create figure
    plt.figure(figsize=(15, 10))

    # Create horizontal bar plot
    sns.barplot(
        data=df_sorted,
        y='Token',
        x='Attention',
        order=df_sorted['Token'],
        color='darkblue'
    )

    plt.title('Mean Attention Weights by Token Position')
    plt.xlabel('Mean Attention Weight')
    plt.ylabel('Token (with position)')

    plt.tight_layout()
    return plt.gcf()

def top_tokens(model_outputs, input_ids, tokenizer, top_k=30):
    """Process and visualize mean attention weights across all layers."""
    attention_weights = model_outputs.attentions

    if attention_weights:
        # Process attention weights
        all_layers_attention = [layer.cpu().float().numpy() for layer in attention_weights]
        stacked_attention = np.stack(all_layers_attention)
        last_token_attention = stacked_attention[:, 0, :, -1, :]
        mean_attention = last_token_attention.mean(axis=(0, 1))

        # Get and filter tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        filtered_tokens, filtered_attention, keep_indices = filter_tokens(tokens, mean_attention, keep_ind=True)

        # Process unique token-position pairs
        token_info = process_unique_tokens(filtered_tokens, filtered_attention, keep_indices)

        # Sort by attention weight
        sorted_info = sorted(token_info, key=lambda x: x[1], reverse=True)

        # Create visualization
        fig = plot_mean_attention(sorted_info, top_k=top_k)

        # # Calculate statistics
        # unique_tokens = len(set(t.split(' (')[0] for t, _, _ in token_info))
        # stats = {
        #     "num_tokens": len(filtered_tokens),
        #     "num_unique_tokens": unique_tokens,
        #     "max_attention": filtered_attention.max(),
        #     "min_attention": filtered_attention.min(),
        #     "mean_attention": filtered_attention.mean()
        # }

        return pd.DataFrame(sorted_info), fig
    else:
        print("Unable to get attention weights")
        return None, None, None
    
