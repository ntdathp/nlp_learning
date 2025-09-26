#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS_5 = ["very_negative","negative","neutral","positive","very_positive"]
LBL2ID = {l:i for i,l in enumerate(LABELS_5)}
ID2LBL = {i:l for l,i in LBL2ID.items()}

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
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def load_texts_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines, None  # no labels

def load_texts_from_csv(path):
    df = pd.read_csv(path)
    assert "text" in df.columns, "CSV phải có cột 'text'"
    texts = df["text"].astype(str).tolist()
    labels = None
    if "label" in df.columns:
        labels = [LBL2ID[l] if l in LBL2ID else None for l in df["label"].astype(str)]
    return texts, labels, df

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def apply_preproc(texts, normalize=True, use_seg=False):
    out = []
    for t in texts:
        s = normalize_text(t) if normalize else t
        s = maybe_segment(s, use_seg=use_seg)
        out.append(s)
    return out

def maybe_apply_class_bias(logits, bias_vec):
    # bias_vec: list of floats length=5, added to logits (logit adjustment)
    if bias_vec is None: return logits
    b = np.array(bias_vec, dtype=np.float32).reshape(1, -1)
    return logits + b

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # ----- Load input -----
    texts, labels, df_src = None, None, None
    if args.input_txt:
        texts, _ = load_texts_from_txt(args.input_txt)
    elif args.input_csv:
        texts, labels, df_src = load_texts_from_csv(args.input_csv)
    else:
        print("Cần --input_txt hoặc --input_csv", file=sys.stderr)
        sys.exit(1)

    if len(texts) == 0:
        print("Không có câu nào để dự đoán.", file=sys.stderr)
        sys.exit(1)

    # ----- Preprocess same as train (normalize; no prefix at infer) -----
    texts_proc = apply_preproc(texts, normalize=args.normalize, use_seg=args.use_seg)

    # ----- Predict in batches -----
    all_logits = []
    with torch.no_grad():
        for chunk in batched(texts_proc, args.batch_size):
            enc = tok(chunk, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = mdl(**enc).logits.detach().cpu().numpy()
            all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)

    # optional class-bias (e.g., penalize neutral)
    bias = None
    if args.neutral_penalty != 0.0:
        bias = [0.0, 0.0, args.neutral_penalty, 0.0, 0.0]  # add to logits
    logits = maybe_apply_class_bias(logits, bias)

    probs = softmax(logits)
    pred_ids = probs.argmax(-1)
    pred_labels = [ID2LBL[int(i)] for i in pred_ids]
    pmax = probs.max(axis=1)

    # ----- Console print (pretty) -----
    print("\n=== Predictions (first N) ===")
    show_n = min(len(texts), args.show)
    for t, pi, p in zip(texts[:show_n], pred_ids[:show_n], pmax[:show_n]):
        print(f"{ID2LBL[int(pi)]:14s}  {p:0.3f}  | {t}")
    if len(texts) > show_n:
        print(f"... ({len(texts)-show_n} more)")

    # ----- If labels exist, compute metrics -----
    if labels is not None and any(l is not None for l in labels):
        idx = [i for i,l in enumerate(labels) if l is not None]
        y_true = np.array([labels[i] for i in idx], dtype=int)
        y_pred = pred_ids[idx]
        print("\n=== Metrics (on rows with valid labels) ===")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Macro F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=LABELS_5, digits=4))

    # ----- Save CSV if requested -----
    if args.out_csv:
        if df_src is None:
            df_out = pd.DataFrame({"text": texts})
        else:
            df_out = df_src.copy()
        df_out["_pred"] = pred_labels
        df_out["_pmax"] = pmax
        for i, name in enumerate(LABELS_5):
            df_out[f"prob_{name}"] = probs[:, i]
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
        print(f"\n[Saved] {args.out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/home/dat/llm_ws/phobert_5cls_clean", help="Thư mục model đã save_model()")
    # input options
    ap.add_argument("--input_txt", default="", help="File .txt: mỗi dòng 1 câu")
    ap.add_argument("--input_csv", default="/home/dat/llm_ws/data/test/vn_product_reviews_test_100_challenge.csv", help="File .csv: bắt buộc có cột 'text', tuỳ chọn 'label'")
    # runtime
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--show", type=int, default=20, help="Số dòng in ra màn hình")
    # preprocessing
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--use_seg", action="store_true", help="Bật word segmentation nếu lúc train có bật")
    # small logit bias (e.g., neutral_penalty=-0.2 để giảm thiên vị neutral)
    ap.add_argument("--neutral_penalty", type=float, default=0.0,
                    help="Giảm/ tăng logit của lớp neutral (âm để phạt neutral). Ví dụ: -0.2")
    # outputs
    ap.add_argument("--out_csv", default="", help="(Tuỳ chọn) Lưu CSV dự đoán + xác suất")
    args = ap.parse_args()
    main(args)
