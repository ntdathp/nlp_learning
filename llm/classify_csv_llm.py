import requests, json, re, sys

MODEL = "qwen2.5:14b-instruct"
URL   = "http://localhost:11434/api/generate"
LABELS = {"very_positive","positive","neutral","negative","very_negative"}

SYSTEM_PROMPT = (
    "Bạn là bộ phân loại cảm xúc tiếng Việt.\n"
    "Chỉ trả lời đúng MỘT JSON duy nhất dạng: {\"label\":\"<một trong 5 nhãn>\"}\n"
    "Năm nhãn hợp lệ: very_positive, positive, neutral, negative, very_negative.\n"
    "Không giải thích, không thêm chữ nào ngoài JSON."
)

def classify(text: str) -> str:
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nPhân loại câu sau:\n\"{text}\"\nJSON:"
    )
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        # 👇 Quan trọng: unload model ngay khi xong
        "keep_alive": 0,
        "options": {"temperature": 0, "top_p": 1, "seed": 42, "num_ctx": 2048}
    }
    r = requests.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json().get("response","").strip()

    m = re.search(r'\{.*\}', out, flags=re.DOTALL)
    if not m:
        return "neutral"

    try:
        obj = json.loads(m.group(0))
        label = str(obj.get("label","")).strip().lower().replace(" ", "_")
        return label if label in LABELS else "neutral"
    except Exception:
        return "neutral"

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Dùng văn phòng thì được , vỏ máy khá"
    print(classify(text))
