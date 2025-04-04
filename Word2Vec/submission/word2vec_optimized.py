import os
import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import argparse

from pathlib import Path


parser = argparse.ArgumentParser(description="Train a Word2Vec model")

parser.add_argument('--data_path', type=str, default='../data/hate/train.csv', help='Path to training CSV')
parser.add_argument('--save_dir', type=str, default='models/word2vec', help='Directory to save model and embeddings')
parser.add_argument('--embedding_dim', type=int, default=100, help='Size of word embeddings')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--window_size', type=int, default=3, help='Window size for context')
parser.add_argument('--neg_samples', type=int, default=3, help='Number of negative samples')

args = parser.parse_args()



# Create dataset-specific subfolder
dataset_name = Path(args.data_path).parent.name  # Gets "hate" from "data/hate/train.csv"
save_subdir = os.path.join(args.save_dir, dataset_name)
os.makedirs(save_subdir, exist_ok=True)

# ===============================
# 1️⃣ Load Dataset & Create Vocabulary
# ===============================
df = pd.read_csv(args.data_path)


def clean_text(x):
    x = re.sub(r'http\S+|www\S+|https\S+', '', str(x))
    x = re.sub(r'\s+', ' ', x)
    x = re.sub(r'[^a-zA-Z0-9\s]', '', x)
    x = re.sub(r'\d+', '', x)
    x = re.sub(r'[^\w\s]', '', x)
    x = re.sub(r'#', '', x)
    x = re.sub(r'\s+', ' ', x)
    x = x.strip()
    x = x.lower()
    return x

df['Sentence'] = df['Sentence'].apply(clean_text)

# Create Vocabulary
vocab = Counter(word for sentence in df['Sentence'] for word in sentence.split())
word2idx = {word: i for i, (word, _) in enumerate(vocab.items())}
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(word2idx)
print("Vocabulary size:", vocab_size)






# Convert Sentences to Word Indices
train_sentences = [word2idx[word] for sentence in df['Sentence'] for word in sentence.split()]
train_sentences = np.array(train_sentences, dtype=np.int32)

# ===============================
# 2️⃣ Optimized Training Pair Generation
# ===============================

WINDOW_SIZE = args.window_size
NUM_NEGATIVE_SAMPLES = args.neg_samples
NEGATIVE_SAMPLING_TABLE_SIZE = 1_000_000  # Large table for efficient sampling

# **🔹 Precompute a large table for negative sampling**
word_freqs = np.array(list(vocab.values()), dtype=np.float32)
word_freqs = word_freqs ** 0.75  # Smooth frequency
word_freqs /= word_freqs.sum()  # Normalize
negative_sample_table = np.random.choice(
    list(word2idx.values()), size=NEGATIVE_SAMPLING_TABLE_SIZE, p=word_freqs
)

# **🔹 Generate Training Pairs Efficiently**
data = []
neg_idx = 0

for i in tqdm.tqdm(range(WINDOW_SIZE, len(train_sentences) - WINDOW_SIZE), desc="Generating Pairs"):
    center_word = train_sentences[i]
    context_words = np.concatenate((train_sentences[i - WINDOW_SIZE:i], train_sentences[i + 1:i + 1 + WINDOW_SIZE]))

    for context_word in context_words:
        data.append([center_word, context_word, 1])  # Positive sample

        # **Fast Negative Sampling**
        neg_samples = []
        while len(neg_samples) < NUM_NEGATIVE_SAMPLES:
            neg_word = negative_sample_table[neg_idx]
            neg_idx = (neg_idx + 1) % NEGATIVE_SAMPLING_TABLE_SIZE  # Circular index
            if neg_word not in context_words and neg_word != center_word:
                neg_samples.append(neg_word)

        for neg_word in neg_samples:
            data.append([center_word, neg_word, 0])  # Negative sample

# Convert to DataFrame
df_pairs = pd.DataFrame(data, columns=['center_word', 'context_word', 'label'])
df_pairs = df_pairs.drop_duplicates().reset_index(drop=True)

print(f"Total training pairs: {len(df_pairs)}")


# ===============================
# 3️⃣ PyTorch Dataset & DataLoader
# ===============================

class Word2VecDataset(Dataset):
    def __init__(self, df):
        self.center_words = torch.tensor(df['center_word'].values, dtype=torch.long)
        self.context_words = torch.tensor(df['context_word'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.center_words)

    def __getitem__(self, idx):
        return self.center_words[idx], self.context_words[idx], self.labels[idx]

# Create DataLoader
BATCH_SIZE = args.batch_size
dataset = Word2VecDataset(df_pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# 4️⃣ Define Word2Vec Model
# ===============================

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, target, context):
        target_emb = self.embeddings(target)  # (batch_size, embed_dim)
        context_emb = self.output_embeddings(context)  # (batch_size, embed_dim)
        scores = torch.sum(target_emb * context_emb, dim=1)  # Dot product similarity
        return scores

# ===============================
# 5️⃣ Training Loop
# ===============================

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model Configuration
EMBEDDING_DIM = args.embedding_dim
EPOCHS = args.epochs
LR = args.lr
# Initialize Model
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()  # More stable than BCELoss
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for center, context, labels in loop:
        center, context, labels = center.to(device), context.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(center, context)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss / len(dataloader)}")

# ===============================
# 6️⃣ Save Model
# ===============================
os.makedirs(args.save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_subdir, f"word2vec_{args.epochs}epoch_cleantext.pth"))
print("✅ Model saved successfully!")

import numpy as np

# Extract word embeddings
word_embeddings = model.embeddings.weight.data.cpu().numpy()  # Move to CPU if on GPU

# Save embeddings with word mappings
embedding_dict = {idx2word[i]: word_embeddings[i] for i in range(vocab_size)}
np.save(os.path.join(save_subdir, f"word_embeddings_{args.epochs}epochs_cleantext.npy"), embedding_dict)



print("✅ Word embeddings saved successfully!")