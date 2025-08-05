import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import re
from tqdm import tqdm
import joblib


# Функция для извлечения лингвистических признаков
def extract_text_features(text):
    words = re.findall(r'\w+', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    word_counts = Counter(words)
    num_words = len(words)
    num_sentences = len(sentences)

    return {
        'word_count': num_words,
        'unique_word_ratio': len(word_counts) / num_words if num_words > 0 else 0,
        'avg_word_length': sum(len(w) for w in words) / num_words if num_words > 0 else 0,
        'avg_sentence_length': num_words / num_sentences if num_sentences > 0 else 0,
        'punct_ratio': len(re.findall(r'[.,;:!?]', text)) / num_words if num_words > 0 else 0,
        'exclam_ratio': text.count('!') / num_words if num_words > 0 else 0
    }


def load_data(pairs_path, truth_path):
    """Загрузка данных с прогресс-баром"""
    texts, labels = [], []
    with open(pairs_path, 'r', encoding='utf-8') as f_pairs, \
            open(truth_path, 'r', encoding='utf-8') as f_truth:
        for line_pair, line_truth in tqdm(zip(f_pairs, f_truth), desc="Loading data", unit=" pairs"):
            pair_data = json.loads(line_pair)
            truth_data = json.loads(line_truth)
            if pair_data["id"] == truth_data["id"]:
                texts.append(pair_data["pair"])
                labels.append(truth_data["same"])
    return texts, labels

print("Loading data...")
texts, labels = load_data(
    "pan20-authorship-verification-test/pan20-authorship-verification-test.jsonl",
    "pan20-authorship-verification-test/pan20-authorship-verification-test-truth.jsonl"
)

#TF-IDF
print("\nPreparing TF-IDF features...")
all_texts = [text for pair in tqdm(texts, desc="Processing texts") for text in pair]
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
tfidf.fit(all_texts)


def create_features(text1, text2, tfidf):
    """Создание расширенного набора признаков"""
    #TF-IDF схожесть
    vec1 = tfidf.transform([text1])
    vec2 = tfidf.transform([text2])
    cosine_sim = cosine_similarity(vec1, vec2)[0][0]

    #Лингвистические признаки
    feats1 = extract_text_features(text1)
    feats2 = extract_text_features(text2)

    #Комбинирование признаков
    return [
        cosine_sim,
        abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1),
        abs(feats1['word_count'] - feats2['word_count']),
        abs(feats1['unique_word_ratio'] - feats2['unique_word_ratio']),
        abs(feats1['punct_ratio'] - feats2['punct_ratio']),
        abs(feats1['exclam_ratio'] - feats2['exclam_ratio']),
        feats1['word_count'],  # Абсолютные значения
        feats2['word_count'],
        feats1['avg_word_length'],
        feats2['avg_word_length']
    ]


#матрица признаков
print("\nCreating feature matrix...")
X = np.array([create_features(pair[0], pair[1], tfidf) for pair in tqdm(texts, desc="Processing pairs")])
y = np.array(labels)

#нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Балансировка классов
print("\nClass distribution before balancing:", Counter(y))
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("Class distribution after balancing:", Counter(y_res))

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

# Оценка
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Важность признаков
print("\nFeature Importance:")
feature_names = [
    'tfidf_cosine', 'length_diff', 'word_count_diff',
    'unique_word_diff', 'punct_ratio_diff', 'exclam_diff',
    'text1_word_count', 'text2_word_count',
    'text1_avg_word_len', 'text2_avg_word_len'
]

for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Сохранение всех компонентов
components = {
    'model': model,  # Модель
    'tfidf': tfidf,  # TfidfVectorizer
    'scaler': scaler  # StandardScaler
}

joblib.dump(components, 'author_verification_components.pkl')

print("\nAll components saved to author_verification_components.pkl")
