# Spam Email Classifier

這是一個垃圾郵件分類專案，使用 Python 和機器學習技術來區分垃圾郵件（spam）和非垃圾郵件（ham）。

## 功能

- **數據預處理:** 清理和標準化郵件文本。
- **模型訓練:** 使用 Logistic Regression 訓練分類模型。
- **模型評估:** 產生混淆矩陣、ROC 曲線和 PR 曲線等視覺化報告。
- **Web 應用:** 提供一個 Streamlit 應用程式，用於即時預測和模型性能分析。

## 專案結構

```
spamEmail/
├── app/                  # Streamlit 應用程式
│   └── streamlit_app.py
├── datasets/             # 數據集
│   ├── sms_spam_no_header.csv
│   └── processed/
├── models/               # 訓練好的模型
│   ├── spam_logreg_model.joblib
│   └── spam_tfidf_vectorizer.joblib
├── reports/              # 評估報告和視覺化圖表
├── scripts/              # 腳本
│   ├── preprocess_emails.py
│   ├── train_spam_classifier.py
│   └── predict_spam.py
└── requirements.txt      # 相依性列表
```

## 安裝

1.  複製此專案：
    ```bash
    git clone <repository-url>
    ```
2.  安裝相依性：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

### 訓練模型

```bash
python scripts/train_spam_classifier.py
```

### 執行 Streamlit 應用程式

```bash
streamlit run app/streamlit_app.py
```

### 進行預測

可以透過 Streamlit 應用程式進行即時預測，或使用以下腳本：

```bash
python scripts/predict_spam.py "Your email content here"
```

## 相依性

- scikit-learn
- pandas
- numpy
- scipy
- joblib
- matplotlib
- seaborn
- streamlit
