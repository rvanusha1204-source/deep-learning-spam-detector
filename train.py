"""Deep Learning Spam Email/SMS Detection (PyTorch Bi-LSTM).  Run: python train.py"""
import os, json, urllib.request
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from common import clean, encode, SpamNet, MAX_LEN, MAX_WORDS

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = "data/sms.tsv"          # label<TAB>text  (ham / spam)
for d in ("data", "model", "outputs"): os.makedirs(d, exist_ok=True)

# 1. Data
if not os.path.exists(DATA_PATH):
    print("Downloading dataset..."); urllib.request.urlretrieve(DATA_URL, DATA_PATH)
df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"]).dropna()
df["y"] = (df["label"] == "spam").astype(int)
print(df["label"].value_counts())
df["clean"] = df["text"].apply(clean)
Xtr_t, Xte_t, ytr, yte = train_test_split(df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=SEED)

# 2. Baseline
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
nb = MultinomialNB().fit(tf.fit_transform(Xtr_t.apply(clean)), ytr)
print("\n=== Baseline Naive Bayes ===")
nbp = nb.predict(tf.transform(Xte_t.apply(clean)))
print("Accuracy:", round(accuracy_score(yte, nbp), 4))
print(classification_report(yte, nbp, target_names=["Ham", "Spam"]))

# 3. Vocabulary + tensors
cnt = Counter(w for t in Xtr_t for w in clean(t).split())
vocab = {w: i + 2 for i, (w, _) in enumerate(cnt.most_common(MAX_WORDS - 2))}
def tens(texts): return torch.tensor([encode(t, vocab) for t in texts])
Xtr, Xte = tens(Xtr_t), tens(Xte_t)
ytr_t, yte_t = torch.tensor(ytr.values, dtype=torch.float32), torch.tensor(yte.values, dtype=torch.float32)
n_val = int(0.1 * len(Xtr))
perm = torch.randperm(len(Xtr))
vi, ti = perm[:n_val], perm[n_val:]
train_dl = DataLoader(TensorDataset(Xtr[ti], ytr_t[ti]), batch_size=32, shuffle=True)
Xval, yval = Xtr[vi], ytr_t[vi]

# 4. Train
model = SpamNet(len(vocab) + 2)
pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best, wait, hist = 1e9, 0, {"tl": [], "vl": [], "va": []}
for ep in range(1, 21):
    model.train(); tl = 0
    for xb, yb in train_dl:
        opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); tl += loss.item() * len(xb)
    model.eval()
    with torch.no_grad():
        lg = model(Xval); vl = loss_fn(lg, yval).item(); va = ((lg > 0).float() == yval).float().mean().item()
    hist["tl"].append(tl / len(ti)); hist["vl"].append(vl); hist["va"].append(va)
    print(f"Epoch {ep:2d}  train_loss {hist['tl'][-1]:.4f}  val_loss {vl:.4f}  val_acc {va:.4f}")
    if vl < best: best, wait = vl, 0; torch.save(model.state_dict(), "model/spam_model.pt")
    else:
        wait += 1
        if wait >= 3: print("Early stopping"); break
model.load_state_dict(torch.load("model/spam_model.pt"))

# 5. Evaluate
model.eval()
with torch.no_grad(): pred = (model(Xte) > 0).int().numpy()
print("\n=== Bi-LSTM (PyTorch) ===")
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print(classification_report(yte, pred, target_names=["Ham", "Spam"]))
plt.figure(figsize=(4, 3))
sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist["tl"], label="train"); ax[0].plot(hist["vl"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["va"]); ax[1].set_title("Validation accuracy")
plt.tight_layout(); plt.savefig("outputs/training_curves.png", dpi=150); plt.close()
json.dump({"vocab": vocab, "vocab_size": len(vocab) + 2}, open("model/vocab.json", "w"))
print("Saved model/ and outputs/")
