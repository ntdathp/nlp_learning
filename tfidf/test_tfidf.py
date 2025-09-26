# test.py
from joblib import load

# Nếu dùng underthesea:
def analyzer(doc):
    try:
        from underthesea import word_tokenize
        return word_tokenize(str(doc), format="text").split()
    except Exception:
        return str(doc).split()

model = load("tfidf_baseline.joblib")

samples = [
    "máy ảnh nét, màu sắc ổn định, rất ưng",
    "giao hàng chậm, sản phẩm trầy xước",
    "Mình thấy laptop bình thường",
    "Sản phẩm tạm ổn"
]
print(model.predict(samples))
