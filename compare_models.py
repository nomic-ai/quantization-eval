import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

dirs = glob.glob('model_outputs/test-run-*')

plt.figure(figsize=(15, 8))

precision_order = ['full', 'bnb8bit', 'bnb4bit-nf4', 'bnb4bit']
precision_labels = {
    'full': 'Precision 16bit',
    'bnb8bit': 'Precision 8bit',
    'bnb4bit-nf4': 'Precision 4bit-nf4',
    'bnb4bit': 'Precision 4bit'
}

for precision_key in precision_order:
    if precision_key == 'bnb4bit':
        # For bnb4bit, explicitly exclude nf4
        dir_path = next(d for d in dirs if 'bnb4bit' in d and 'nf4' not in d)
    else:
        dir_path = next(d for d in dirs if precision_key in d)
    csv_path = os.path.join(dir_path, 'top_tokens.csv')
    print(csv_path)
    df = pd.read_csv(csv_path, header=0, names=['token', 'attention', 'position'])
    df['token'] = df['token'].str.replace(r'\s*\(pos \d+(?:, occ \d+)?\)', '', regex=True)
    df = df.sort_values('position')
    plt.plot(df['position'], df['attention'], label=precision_labels[precision_key], marker='o')
    if precision_key == precision_order[0]:
        plt.xticks(df['position'], df['token'], rotation=45, ha='right')

plt.title('Token Attention Scores by Position in Prompt')
plt.xlabel('Tokens (in original order)')
plt.ylabel('Attention Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('token_comparison.png')
plt.show()