# train_tfidf_baseline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# --------- Load data ----------
df = pd.read_csv("/home/dat/llm_ws/data/train/vn_product_reviews_5sentiments_10k_v2_clean.csv")  # hoặc file synthetic bạn đã tạo
df = df.dropna(subset=["text","label"])
X = df["text"].astype(str).tolist()
y = df["label"].astype(str).tolist()

# --------- Split giữ tỉ lệ lớp ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------- Class weights (nếu lệch lớp) ----------
classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw_map = {c:w for c,w in zip(classes, class_weights)}

# --------- Pipeline TF-IDF + (model) ----------
# Không dùng analyzer callable => ngram_range hoạt động bình thường
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1,2),
        min_df=2,
        # token_pattern mặc định bỏ từ 1 ký tự; dùng pattern dưới để giữ cả 1 ký tự & Unicode
        token_pattern=r"(?u)\b\w+\b",
        # strip_accents giúp chuẩn hoá dấu nếu cần:
        # strip_accents="unicode",
    )),
    ("clf", LogisticRegression(max_iter=2000))  # placeholder, sẽ set qua GridSearch
])

# --------- GridSearch: Logistic vs LinearSVC ----------
param_grid = [
    {
        "clf": [LogisticRegression(max_iter=2000, class_weight=cw_map, solver="liblinear")],
        "tfidf__max_df": [0.9, 1.0],
        "tfidf__sublinear_tf": [True, False],
        "clf__C": [0.5, 1.0, 2.0]
    },
    {
        "clf": [LinearSVC(class_weight=cw_map, max_iter=5000)],
        "tfidf__max_df": [0.9, 1.0],
        "tfidf__sublinear_tf": [True, False],
        "clf__C": [0.5, 1.0, 2.0]
        # Với LinearSVC mới, có thể thêm "clf__dual": ["auto", True] nếu muốn
    }
]

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV score (f1_macro):", gs.best_score_)

# --------- Đánh giá test ----------
y_pred = gs.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=classes))

# --------- Lưu model ----------
dump(gs.best_estimator_, "tfidf_baseline.joblib")
print("\nSaved model to tfidf_baseline.joblib")
