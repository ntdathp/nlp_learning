#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS_5 = ["very_negative","negative","neutral","positive","very_positive"]
ID2LBL = {i:l for i,l in enumerate(LABELS_5)}

# ==== same normalize as training ====
EMO_POS = ["🤩","🥰","😍","❤️","👍","😎","👌","✨","🔥","💯"]
EMO_NEG = ["😱","😡","🤬","💩","👎","😤","😞","😭"]

def normalize_text(s: str) -> str:
    s = str(s).strip()
    for e in EMO_POS: s = s.replace(e, " EMO_POS ")
    for e in EMO_NEG: s = s.replace(e, " EMO_NEG ")
    repl = {
        "vl": "rất", "okeee": "ok", "ưng": "rất thích",
        "siêu siêu": "rất", "siêu thất vọng": "rất thất vọng",
        "mãi đỉnh": "rất tốt", "best of best": "rất tốt", "best choice": "rất tốt",
        "đỉnh của chóp": "rất tốt",
    }
    for k,v in repl.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
    return s

def maybe_segment(text, use_seg=False):
    if not use_seg: return text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        # Không có underthesea thì bỏ qua segmentation
        return text

def softmax(x):
    x = x - np.max(x, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, keepdims=True)

def main():
    ap = argparse.ArgumentParser(description="Quick inference cho 1 câu (PhoBERT 5 lớp).")
    ap.add_argument("--model_dir", default="/home/dat/llm_ws/phobert_5cls_clean",
                    help="Thư mục model đã save_model()")
    ap.add_argument("--text", required=True, help="Câu cần dự đoán")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--use_seg", action="store_true", help="Bật word segmentation nếu lúc train có bật")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--neutral_penalty", type=float, default=0.0,
                    help="Điều chỉnh logit lớp neutral (âm để phạt neutral), ví dụ -0.2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # Preprocess
    s = args.text
    if args.normalize:
        s = normalize_text(s)
    s = maybe_segment(s, use_seg=args.use_seg)

    # Encode & forward
    with torch.no_grad():
        enc = tok([s], truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits.detach().cpu().numpy()[0]

    # Optional bias vào lớp neutral
    if args.neutral_penalty != 0.0:
        logits[2] += args.neutral_penalty  # index 2 là 'neutral'

    probs = softmax(logits)
    pred_id = int(probs.argmax())
    pred_lbl = ID2LBL[pred_id]

    # In kết quả
    print("Text:", args.text)
    print("Pred:", pred_lbl)
    print("Conf:", f"{float(probs[pred_id]):.6f}")
    print("Probs:")
    for i, name in enumerate(LABELS_5):
        print(f"  {name:15s}: {float(probs[i]):.6f}")

if __name__ == "__main__":
    main()
