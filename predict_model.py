import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# Функция для чтения текста из текстового файла
def read_text_from_file(file_path):
    """Чтение текста из текстового файла"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Функция для записи текста в файл
def write_text_to_file(text, output_path):
    """Запись текста в файл"""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Функция для извлечения лингвистических признаков
def extract_text_features(text):
    """Извлечение лингвистических признаков"""
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

# Функция для создания признаков
def create_features(text1, text2, tfidf):
    """Создание расширенного набора признаков"""
    # TF-IDF схожесть
    vec1 = tfidf.transform([text1])
    vec2 = tfidf.transform([text2])
    cosine_sim = cosine_similarity(vec1, vec2)[0][0]

    # Лингвистические признаки
    feats1 = extract_text_features(text1)
    feats2 = extract_text_features(text2)

    # Комбинируем все признаки в плоский массив
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

# Загрузка всех компонентов
components = joblib.load('author_verification_components.pkl')

# Извлечение компонентов
model = components['model']  # Модель RandomForestClassifier
tfidf = components['tfidf']  # TfidfVectorizer
scaler = components['scaler']  # StandardScaler

# Пути к входным текстовым файлам
file1_path = "texts/1txt.txt"  # Замените на путь к вашему первому текстовому файлу
file2_path = "texts/2txt.txt"  # Замените на путь к вашему второму текстовому файлу

# Чтение текста из файлов
text1 = read_text_from_file(file1_path)
text2 = read_text_from_file(file2_path)

# Сохранение текстов в новые файлы для проверки
output_file1 = "analyzed_text1.txt"
output_file2 = "analyzed_text2.txt"

write_text_to_file(text1, output_file1)
write_text_to_file(text2, output_file2)

print(f"Текст из первого файла сохранён в {output_file1}")
print(f"Текст из второго файла сохранён в {output_file2}")

# Создание признаков
features = np.array([create_features(text1, text2, tfidf)])

# Нормализация признаков
features = scaler.transform(features)

# Предсказание
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

# Вывод результатов
if prediction == 1:
    print("Тексты написаны одним автором.")
else:
    print("Тексты написаны разными авторами.")

print(f"Вероятность того, что тексты написаны одним автором: {probability:.4f}")