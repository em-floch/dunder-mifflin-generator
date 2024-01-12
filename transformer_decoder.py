import torch
import torch.nn as nn
import torch.nn.functional as F



class Head(nn.Module):

    def __init__(self, head_size, n_emd_dim, block_size, drop_rate):
        super().__init__()
        self.key = nn.Linear(n_emd_dim, head_size, bias=False)
        self.query = nn.Linear(n_emd_dim, head_size, bias=False)
        self.value = nn.Linear(n_emd_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        x = wei @ v
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size, n_emd_dim, block_size, drop_rate):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_emd_dim, block_size, drop_rate) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_emd_dim)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(out)



class FeedForward(nn.Module):

    def __init__(self, n_emd_dim, drop_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emd_dim,  4 * n_emd_dim),
            nn.ReLU(),
            nn.Linear(4 * n_emd_dim,  n_emd_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):

    def __init__(self, n_emd_dim, n_head, block_size, drop_rate):
        super().__init__()
        head_size = n_emd_dim // n_head
        self.attention_head = MultiHeadAttention(n_head, head_size, n_emd_dim, block_size, drop_rate)
        self.feed_forward = FeedForward(n_emd_dim, drop_rate)
        self.ln1 = nn.LayerNorm(n_emd_dim)
        self.ln2 = nn.LayerNorm(n_emd_dim)

    def forward(self, x):
        x = x + self.attention_head(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x



class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_emd_dim, block_size, n_layer, n_head, drop_rate):
        super().__init__()

        self.block_size = block_size

        self.embedding_table = nn.Embedding(vocab_size, n_emd_dim)
        self.positional_embedding = nn.Embedding(block_size, n_emd_dim)
        self.blocks = nn.Sequential(*[Block(n_emd_dim, n_head, block_size, drop_rate) for _ in range(n_layer)],
                                    )
        self.ln = nn.LayerNorm(n_emd_dim)
        self.lm_head = nn.Linear(n_emd_dim, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_emb = self.embedding_table(idx)
        position_emb = self.positional_embedding(torch.arange(T))
        x_emb = token_emb + position_emb
        x = self.blocks(x_emb)
        x = self.ln(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_tokens=100):
        for i in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

