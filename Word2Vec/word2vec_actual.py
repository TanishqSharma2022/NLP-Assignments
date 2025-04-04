#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('jupyter nbconvert --to script word2vec_actual.ipynb')


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

import tqdm


# In[5]:


df = pd.read_csv('../data/hate/train.csv')
val_hate = pd.read_csv('../data/hate/val.csv')


# # `Creating Vocabulary`

# In[6]:


vocab = []

for sentence in df['Sentence']:
    words = sentence.split()
    vocab.extend(words)
    
wordCount = Counter(vocab)
vocab = {word: i for i, (word, _) in enumerate(wordCount.items())}
inv_vocab = {i: word for word, i in vocab.items()}  # Reverse mapping
vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)


# In[7]:


train_sentences = []

for sentence in df['Sentence']:
    train_sentences.extend(sentence.split())

len(train_sentences)


# In[ ]:


train_sentences = train_sentences


# In[9]:


WINDOW_SIZE = 3
NUM_NEGATIVE_SAMPLES = 3

data = []

# Iterate over all words with tqdm progress bar
for idx, center_word in tqdm.tqdm(enumerate(train_sentences[WINDOW_SIZE-1:-WINDOW_SIZE]), 
                             total=len(train_sentences) - 2 * (WINDOW_SIZE - 1), 
                             desc="Processing Pairs"):

    # Get context words around the center word
    context_words = [context_word for context_word in train_sentences[idx:idx+2*WINDOW_SIZE-1] if context_word != center_word]
    
    for context_word in context_words:
        data.append([center_word, context_word, 1])  # Positive pair
        
        # Get negative samples (words NOT in current context)
        negative_samples = np.random.choice(
            [w for w in train_sentences[WINDOW_SIZE-1:-WINDOW_SIZE] if w != center_word and w not in context_words], 
            NUM_NEGATIVE_SAMPLES, 
            replace=False
        )

        for negative_samp in negative_samples:
            data.append([center_word, negative_samp, 0])  # Negative pair


# In[10]:


df = pd.DataFrame(columns=['center_word', 'context_word', 'label'], data=data)
words = np.intersect1d(df.context_word, df.center_word)
df = df[(df.center_word.isin(words)) & (df.context_word.isin(words))].reset_index(drop=True)
df


# In[20]:


len(vocab)


# In[11]:


import pandas as pd

# Create a new DataFrame (deep copy to avoid modifying the original)
df_ids = df.copy()

# Replace words with their corresponding IDs from vocab
df_ids['center_word'] = df_ids['center_word'].map(lambda w: vocab.get(w, -1))
df_ids['context_word'] = df_ids['context_word'].map(lambda w: vocab.get(w, -1))



# In[12]:


# Print the first few rows to verify
df_ids.head()


# In[13]:


import torch
import torch.nn as nn
import torch.optim as optim

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)  # Word embedding matrix
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)  # Context embedding matrix
    
    def forward(self, target, context):
        target_emb = self.embeddings(target)  # (batch_size, embed_dim)
        context_emb = self.output_embeddings(context)  # (batch_size, embed_dim)
        scores = torch.sum(target_emb * context_emb, dim=1)  # Dot product similarity
        return scores

# Model parameters
embed_dim = 100  # Word embedding size
model = Word2Vec(vocab_size, embed_dim)


# In[14]:


X = df_ids[['center_word', 'context_word']].values  # Word ID pairs as input
y = df_ids['label'].values  # Labels (0 or 1)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model Configuration
VOCAB_SIZE = len(vocab) + 1  # Add 1 for unknown words
EMBEDDING_DIM = 100  # As required
HIDDEN_DIM = 64  # Hidden layer size
BATCH_SIZE = 128  # Adjustable
EPOCHS = 50  # Number of training epochs
LR = 0.001  # Learning rate

# Define Word2Vec Model with Binary Classification
class Word2VecClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Word2VecClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Word embedding layer
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)  # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer (binary classification)
        self.sigmoid = nn.Sigmoid()  # Binary classification activation

    def forward(self, x):
        word1 = self.embedding(x[:, 0])  # Get embedding for center_word
        word2 = self.embedding(x[:, 1])  # Get embedding for context_word
        combined = torch.cat((word1, word2), dim=1)  # Concatenate embeddings
        hidden = torch.relu(self.fc1(combined))  # Hidden layer with ReLU
        output = self.sigmoid(self.fc2(hidden))  # Binary classification output
        return output

# Initialize Model and Move to GPU
model = Word2VecClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)


# In[16]:


# Convert DataFrame to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.long).to(device)  # Word indices must be long
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)  # Labels must be float

# Create Dataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[17]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should print True if GPU is available


# In[18]:


# Loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)  # tqdm progress bar
    
    for batch in loop:
        x_batch, y_batch = batch  # Unpack batch
        optimizer.zero_grad()  # Reset gradients
        outputs = model(x_batch).squeeze()  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()

        # Update tqdm bar with loss
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss / len(dataloader)}")



# In[ ]:


torch.save(model.state_dict(), "models/2/word2vec_classifier_50epoch.pth")
print("Model saved successfully!")


# In[ ]:




