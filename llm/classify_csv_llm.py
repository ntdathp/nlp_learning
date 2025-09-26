import requests, json, re, sys

MODEL = "qwen2.5:7b-instruct"          # hoặc "llama3.1:8b-instruct"
URL   = "http://localhost:11434/api/generate"
LABELS = {"very_positive","positive","neutral","negative","very_negative"}

SYSTEM_PROMPT = (
    "Bạn là bộ phân loại cảm xúc tiếng Việt. "
    "Năm nhãn hợp lệ (chỉ được chọn một): very_positive, positive, neutral, negative, very_negative.\n"
    "Chỉ trả về JSON duy nhất dạng: {\"label\":\"<một trong năm nhãn>\"}"
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
        "options": {"temperature": 0, "top_p": 1, "seed": 42}
    }
    r = requests.post(URL, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json().get("response","").strip()

    # Tóm lấy JSON duy nhất trong phản hồi
    m = re.search(r'\{.*\}', out, flags=re.DOTALL)
    if not m:
        return "neutral"  # fallback an toàn

    try:
        obj = json.loads(m.group(0))
        label = str(obj.get("label","")).strip().lower().replace(" ", "_")
        return label if label in LABELS else "neutral"
    except Exception:
        return "neutral"

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Dùng văn phòng thì được , vỏ máy khá"
    print(classify(text))
