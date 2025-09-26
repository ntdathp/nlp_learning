#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import tensorflow as tf

# ===== default labels nếu không có label_map.json =====
DEFAULT_LABELS = ["negative", "neutral", "positive"]

def load_savedmodel(model_dir):
    obj = tf.saved_model.load(model_dir)                  # load kiểu TF
    infer = obj.signatures.get("serving_default")         # signature mặc định
    if infer is None:
        raise RuntimeError("No 'serving_default' signature found.")
    return infer

def predict_one_tf(infer, id2label, text):
    # tên input phải khớp với tên bạn đặt khi export: Input(name="text", dtype=string)
    outs = infer(text=tf.constant([[text]], dtype=tf.string))
    # lấy tensor đầu ra (thường chỉ có 1)
    y = list(outs.values())[0].numpy()[0]                 # shape [C]
    pred_id = int(np.argmax(y))
    pred_label = id2label[pred_id]
    pred_conf = float(y[pred_id])
    ranked = sorted([(id2label[i], float(p)) for i, p in enumerate(y)],
                    key=lambda x: x[1], reverse=True)
    return pred_label, pred_conf, ranked

def load_label_map(model_dir):
    lm_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(lm_path):
        with open(lm_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        print(f"[INFO] Loaded label_map.json from {lm_path}")
    else:
        id2label = {i: lab for i, lab in enumerate(DEFAULT_LABELS)}
        print(f"[WARN] label_map.json not found. Using default labels: {DEFAULT_LABELS}")
    return id2label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/home/dat/llm_ws/bilstm/bilstm_vn_sentiment_multiclass")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    # đọc label map hoặc dùng mặc định
    id2label = load_label_map(args.model_dir)

    # load kiểu TF SavedModel
    infer = load_savedmodel(args.model_dir)

    label, conf, ranked = predict_one_tf(infer, id2label, args.text)

    print(f"\nText : {args.text}")
    print(f"Pred : {label} ({conf:.3f})")
    print("Probs:")
    for lab, p in ranked:
        print(f"  - {lab:14s}: {p:.4f}")

if __name__ == "__main__":
    main()
