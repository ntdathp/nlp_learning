import os, json, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ==== Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ==== Config
CSV_PATH = "/home/dat/llm_ws/data/train/vn_product_reviews_5sentiments_10k_v2_clean.csv"  # <-- dùng bản V2
MAX_VOCAB = 20000
SEQ_LEN   = 64
EMBED_DIM = 128
BILSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None   # ví dụ: {class_id: alpha, ...} hoặc None

# ==== 1) Load data
df = pd.read_csv(CSV_PATH)
assert {"text","label"}.issubset(df.columns), "CSV phải có cột text,label"
df = df.dropna(subset=["text","label"]).copy()
df["text"] = df["text"].astype(str).str.strip()

five_class = ["very_negative", "negative", "neutral", "positive", "very_positive"]
three_class = ["negative", "neutral", "positive"]

unique_labels = sorted(df["label"].unique().tolist())
if set(five_class).issubset(set(unique_labels)):
    class_list = five_class
else:
    class_list = three_class
    df["label"] = df["label"].replace({
        "very_negative": "negative", "very_positive": "positive"
    })

df = df[df["label"].isin(class_list)].copy()

label2id = {c:i for i,c in enumerate(class_list)}
id2label = {i:c for c,i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

print("Classes:", class_list)
print("Counts:", df["label"].value_counts().to_dict())

# ==== 2) Split
try:
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].values, df["label_id"].values,
        test_size=0.2, random_state=SEED, stratify=df["label_id"].values
    )
except ValueError:
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].values, df["label_id"].values,
        test_size=0.2, random_state=SEED
    )

# ==== 2.1) Class weights
cnt = Counter(y_train)
max_count = max(cnt.values())
class_weight = {cls: max_count / cnt[cls] for cls in cnt}
print("Class weights:", {id2label[k]: round(v,3) for k,v in class_weight.items()})

# ==== 3) Vectorizer
text_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation"
)
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(64))

# ==== 4) Model: SpatialDropout1D + BiLSTM + (AvgPool + MaxPool) concat
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)                                   # [B, L]
x = layers.Embedding(MAX_VOCAB, EMBED_DIM, mask_zero=True)(x) # [B, L, D]
x = layers.SpatialDropout1D(0.2)(x)
x = layers.Bidirectional(layers.LSTM(BILSTM_UNITS, return_sequences=True))(x)
# Dual pooling
avg_pool = layers.GlobalAveragePooling1D()(x)
max_pool = layers.GlobalMaxPooling1D()(x)
x = layers.Concatenate()([avg_pool, max_pool])
x = layers.Dropout(0.35)(x)
x = layers.Dense(192, activation="relu")(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(len(class_list), activation="softmax")(x)

model = models.Model(inputs, outputs)

# ==== 4.1) Sparse Categorical Focal Loss
@tf.function
def sparse_categorical_focal_loss(y_true, y_pred, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    y_true: int32 shape [B], y_pred: probs [B, C]
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    # gather p_t
    idx = tf.stack([tf.range(tf.shape(y_pred)[0]), y_true], axis=1)
    p_t = tf.gather_nd(y_pred, idx)  # [B]

    if isinstance(alpha, dict):
        # per-class alpha
        alpha_vec = tf.constant([alpha.get(i, 1.0) for i in range(y_pred.shape[-1])], dtype=tf.float32)
        alpha_t = tf.gather(alpha_vec, y_true)
    elif isinstance(alpha, (float, int)):
        alpha_t = tf.cast(alpha, tf.float32)
    else:
        alpha_t = 1.0

    loss = - alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return tf.reduce_mean(loss)

# Optimizer & compile
opt = optimizers.Adam(learning_rate=2e-3)
model.compile(optimizer=opt, loss=sparse_categorical_focal_loss, metrics=["accuracy"])
model.summary()

# ==== 5) TF Datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(4096, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==== 6) Callbacks: Macro-F1 tracking, ReduceLROnPlateau, EarlyStopping
class MacroF1Callback(callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.best_f1 = -1.0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_ds, verbose=0)
        y_pred = probs.argmax(axis=1)
        y_true = np.concatenate([y for _, y in self.val_ds], axis=0)
        f1 = f1_score(y_true, y_pred, average="macro")
        logs = logs or {}
        logs["val_macro_f1"] = f1
        print(f"\n[Epoch {epoch+1}] val_macro_f1 = {f1:.4f}")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"\nRestored best weights with val_macro_f1 = {self.best_f1:.4f}")

macro_cb = MacroF1Callback(val_ds)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-5, verbose=1
)
earlystop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[macro_cb, reduce_lr, earlystop],
    verbose=1,
    class_weight=class_weight
)

# ==== 7) Evaluate
val_probs = model.predict(val_ds, verbose=0)
val_pred = val_probs.argmax(axis=1)
y_true = np.concatenate([y for _, y in val_ds], axis=0)

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, val_pred))

print("\nClassification report:")
print(classification_report(
    y_true, val_pred, target_names=class_list, digits=4
))
macro_f1 = f1_score(y_true, val_pred, average="macro")
print(f"\nMacro-F1: {macro_f1:.4f}")

# ==== 8) Dự đoán nhanh
def predict_texts(text_list):
    ds = tf.data.Dataset.from_tensor_slices(text_list).batch(64)
    probs = model.predict(ds, verbose=0)
    idx = probs.argmax(axis=1)
    return [(t, float(probs[i, idx[i]]), id2label[idx[i]]) for i, t in enumerate(text_list)]

demo_texts = [
    "Thiết bị robot hút bụi thất vọng ồn shop phản hồi chậm.",
    "Màn hình tuyệt hảo, không chê vào đâu được.",
    "Sản phẩm tệ hại, hỏng ngay khi mở hộp, yêu cầu hoàn tiền.",
    "Mình thấy laptop đóng gói ổn, dùng tạm được.",
    "Điện thoại mượt mà, pin khoẻ, rất đáng tiền."
]
print("\nDemo predictions:")
for t, p, lab in predict_texts(demo_texts):
    print(f"[{lab:14s}] {p:0.3f} | {t}")

# ==== 9) Export SavedModel + label_map (string -> vectorize -> model -> softmax)
EXPORT_DIR = "bilstm_vn_sentiment_multiclass"
if os.path.exists(EXPORT_DIR):
    import shutil; shutil.rmtree(EXPORT_DIR)

inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
z = text_vectorizer(inp)
# replicate forward pass
z = model.layers[2](z)           # Embedding
z = model.layers[3](z)           # SpatialDropout1D
z = model.layers[4](z)           # BiLSTM
avg_pool = layers.GlobalAveragePooling1D()(z)
max_pool = layers.GlobalMaxPooling1D()(z)
z = layers.Concatenate()([avg_pool, max_pool])
z = model.layers[8](z)           # Dropout
z = model.layers[9](z)           # Dense
z = model.layers[10](z)          # Dropout
z = model.layers[11](z)          # Final Dense
full_model = tf.keras.Model(inp, z)
tf.saved_model.save(full_model, EXPORT_DIR)

with open(os.path.join(EXPORT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print(f"\nSavedModel exported to: {EXPORT_DIR}")
print("Label map:", label2id)
