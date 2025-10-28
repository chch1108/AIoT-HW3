import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ----------------------------
# 設定路徑
# ----------------------------
DATA_PATH = "datasets/processed/sms_spam_clean.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# 讀取資料
# ----------------------------
df = pd.read_csv(DATA_PATH)
X_texts = df["text_clean"].astype(str).fillna("")
y = (df["col_0"].str.lower() == "spam").astype(int).values  # 根據資料 label

# ----------------------------
# 訓練 TF-IDF vectorizer
# ----------------------------
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X_texts)

# ----------------------------
# 訓練 LogisticRegression
# ----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, y)

# ----------------------------
# 存檔
# ----------------------------
joblib.dump(vec, os.path.join(MODELS_DIR, "spam_tfidf_vectorizer.joblib"))
joblib.dump(clf, os.path.join(MODELS_DIR, "spam_logreg_model.joblib"))

print(f"模型與 vectorizer 已存檔到 {MODELS_DIR}/")
