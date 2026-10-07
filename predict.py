"""Usage: python predict.py "Win a free prize now!" """
import sys, json, torch
from common import encode, SpamNet

_cfg = json.load(open("model/vocab.json"))
_model = SpamNet(_cfg["vocab_size"])
_model.load_state_dict(torch.load("model/spam_model.pt"))
_model.eval()

def classify(message: str):
    with torch.no_grad():
        p = torch.sigmoid(_model(torch.tensor([encode(message, _cfg["vocab"])]))).item()
    return ("Spam" if p >= 0.5 else "Not Spam (Ham)"), p

if __name__ == "__main__":
    for m in (sys.argv[1:] or ["Congratulations! You won a free prize. Call 09061701461 now to claim."]):
        label, p = classify(m)
        print(f"{label}  (spam probability {p:.3f})\n  -> {m}")
