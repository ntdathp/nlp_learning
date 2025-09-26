#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, numpy as np, torch, unicodedata, string
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS_5 = ["very_negative","negative","neutral","positive","very_positive"]
ID2LBL = {i:l for i,l in enumerate(LABELS_5)}

EMO_POS = ["🤩","🥰","😍","❤️","👍","😎","👌","✨","🔥","💯"]
EMO_NEG = ["😱","😡","🤬","💩","👎","😤","😞","😭"]

TEENCODE_MAP = {
    r"\bko\b": "không", r"\bk\b": "không", r"\bkh\b": "không",
    r"\bkhong\b": "không", r"\bhong\b": "không",
    r"\bvl\b": "rất", r"\bvcl\b": "rất",
    r"\bqa\b": "quá", r"\bqua\b": "quá", r"\bhqa\b": "hơi quá",
    r"\bok+\b": "ok", r"\boke+\b": "ok", r"\bokeee\b": "ok",
}

ACCENT_MAP = {
    "may":"máy","nong":"nóng","mat":"mát","on":"ổn","kem":"kém","tot":"tốt","te":"tệ",
    "thatvong":"thất vọng","tuyetvoi":"tuyệt vời","tuyet":"tuyệt",
    "hai_long":"hài lòng","hai long":"hài lòng",
    "dang":"đáng","gia":"giá","dat":"đắt","re":"rẻ","ton":"tốn","tien":"tiền","mua":"mua",
    "pin":"pin","am":"âm","thanh":"thanh","am thanh":"âm thanh","am_thanh":"âm thanh",
    "dong":"đóng","goi":"gói","dong goi":"đóng gói","donggoi":"đóng gói",
    "nhanh":"nhanh","cham":"chậm","tre":"trễ","lau":"lâu","som":"sớm","ben":"bền","yeu":"yếu",
    "man":"màn","hinh":"hình","man hinh":"màn hình","man_hinh":"màn hình",
    "sac":"sạc","sac du phong":"sạc dự phòng","sacduphong":"sạc dự phòng",
    "nhiet":"nhiệt","do":"độ","nhiet do":"nhiệt độ","nhiet_do":"nhiệt độ",
    "dich vu":"dịch vụ","dichvu":"dịch vụ","ho tro":"hỗ trợ","ho_tro":"hỗ trợ",
    "phan hoi":"phản hồi","phanhoi":"phản hồi",
    "chat luong":"chất lượng","chatluong":"chất lượng",
    "trai nghiem":"trải nghiệm","trai_nghiem":"trải nghiệm",
    "cau tha":"cẩu thả","cau_tha":"cẩu thả",
    "dep":"đẹp","xau":"xấu","xuat sac":"xuất sắc","xuat_sac":"xuất sắc",
    "qua":"quá","rat":"rất","cuc":"cực","cuc ky":"cực kỳ","cuc_ky":"cực kỳ",
    "khong":"không","khong nen":"không nên",
    "dang tien":"đáng tiền","dang gia":"đáng giá",
    "giong":"giống","mo ta":"mô tả","mo_ta":"mô tả",
    "binh thuong":"bình thường","binh_thuong":"bình thường",
    "thich":"thích","rat thich":"rất thích","rat_thich":"rất thích","ung":"ưng",
}

BIGRAM_HINTS = {
    ("rất","tốt"): 2.0, ("rất","thất"): 1.5, ("rất","đáng"): 1.5,
    ("máy","nóng"): 2.0, ("tốn","tiền"): 2.0, ("âm","thanh"): 2.0,
    ("đóng","gói"): 2.0, ("màn","hình"): 2.0, ("chất","lượng"): 2.0,
    ("trải","nghiệm"): 2.0, ("phản","hồi"): 1.5, ("rất","thích"): 1.5,
}

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s).strip())
    for e in EMO_POS: s = s.replace(e, " EMO_POS ")
    for e in EMO_NEG: s = s.replace(e, " EMO_NEG ")
    s = s.lower()
    for pat, rep in TEENCODE_MAP.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    repl = {
        "vl":"rất","okeee":"ok","ưng":"rất thích","siêu siêu":"rất",
        "siêu thất vọng":"rất thất vọng","mãi đỉnh":"rất tốt",
        "best of best":"rất tốt","best choice":"rất tốt","đỉnh của chóp":"rất tốt",
    }
    for k,v in repl.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
    return re.sub(r"\s+"," ", s).strip()

def maybe_segment(text, use_seg=False):
    if not use_seg: return text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text

def approx_diacritic_ratio(s: str) -> float:
    vowels_with_tone = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    s_low = s.lower()
    cnt_diac = sum(ch in vowels_with_tone for ch in s_low)
    cnt_letters = sum(ch.isalpha() for ch in s_low)
    return (cnt_diac / max(1, cnt_letters))

def strip_accents_simple(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = s.replace("đ","d").replace("Đ","D")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def _choose_variant(token_lc: str, prev_lc: str = None):
    cand = ACCENT_MAP.get(token_lc)
    if cand and prev_lc:
        score = 1.0 + BIGRAM_HINTS.get((prev_lc, cand.split()[0]), 0.0)
        return cand, score
    if cand:
        return cand, 1.0
    return None, 0.0

def restore_diacritics(text: str) -> str:
    tokens = text.split(" ")
    out_tokens, prev_norm = [], None
    for tok in tokens:
        # tách punctuation đầu/cuối
        prefix, suffix, core = "", "", tok
        while core and core[0] in string.punctuation:
            prefix += core[0]; core = core[1:]
        while core and core[-1] in string.punctuation:
            suffix = core[-1] + suffix; core = core[:-1]
        if not core:
            out_tokens.append(prefix + suffix); prev_norm = None; continue
        base = strip_accents_simple(core.lower())
        best, _ = _choose_variant(base, prev_norm)
        replaced = best if best else core
        if core.isupper():
            replaced = replaced.upper()
        elif core[0].isupper():
            replaced = replaced[0].upper() + replaced[1:]
        out_tokens.append(prefix + replaced + suffix)
        prev_norm = replaced.split()[0].lower()
    return " ".join(out_tokens)

def softmax(x):
    x = x - np.max(x, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, keepdims=True)

def main():
    ap = argparse.ArgumentParser(description="Quick inference cho 1 câu (PhoBERT 5 lớp).")
    ap.add_argument("--model_dir", default="/home/dat/llm_ws/phobert_5cls_clean")
    ap.add_argument("--text", required=True)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--use_seg", action="store_true")
    ap.add_argument("--normalize", action="store_true", default=True)
    # Auto-restore: bật mặc định; có thể tắt bằng --no_auto_restore
    ap.add_argument("--no_auto_restore", action="store_true",
                    help="Tắt tự động khôi phục dấu khi phát hiện thiếu dấu.")
    ap.add_argument("--restore_threshold", type=float, default=0.03,
                    help="Ngưỡng tỷ lệ ký tự có dấu để kích hoạt khôi phục.")
    ap.add_argument("--neutral_penalty", type=float, default=0.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    raw = args.text
    s = normalize_text(raw) if args.normalize else raw

    # TỰ ĐỘNG phát hiện & khôi phục dấu (trừ khi tắt bằng --no_auto_restore)
    if not args.no_auto_restore and approx_diacritic_ratio(s) < args.restore_threshold:
        restored = restore_diacritics(s)
        print(f"[INFO] Auto-restore diacritics → \"{restored}\"")
        s = restored

    s = maybe_segment(s, use_seg=args.use_seg)

    with torch.no_grad():
        enc = tok([s], truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits.detach().cpu().numpy()[0]

    if args.neutral_penalty != 0.0:
        logits[2] += args.neutral_penalty  # 'neutral' index 2

    probs = softmax(logits)
    pred_id = int(probs.argmax())
    pred_lbl = ID2LBL[pred_id]

    print("Text:", raw)
    print("Pred:", pred_lbl)
    print("Conf:", f"{float(probs[pred_id]):.6f}")
    print("Probs:")
    for i, name in enumerate(LABELS_5):
        print(f"  {name:15s}: {float(probs[i]):.6f}")

if __name__ == "__main__":
    main()
