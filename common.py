import re
import torch
import torch.nn as nn

MAX_LEN = 60
MAX_WORDS = 10000

def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\b\d{5,}\b", " longnumber ", t)
    t = re.sub(r"[£$€]\s?\d+", " moneytoken ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def encode(text: str, vocab: dict):
    ids = [vocab.get(w, 1) for w in clean(text).split()][:MAX_LEN]   # 0 = pad, 1 = unknown
    return ids + [0] * (MAX_LEN - len(ids))

class SpamNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop_e = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden * 2, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        e = self.drop_e(self.emb(x))
        o, _ = self.lstm(e)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (o * mask).sum(1) / mask.sum(1).clamp(min=1)    # masked mean pooling
        h = torch.relu(self.fc1(self.drop(pooled)))
        return self.out(h).squeeze(-1)                            # logit
