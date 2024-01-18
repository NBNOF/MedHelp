import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загружаем данные из JSON файла
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Создаем DataFrame
symptoms_list = []
disease_list = []
for disease, symptoms in data.items():
    symptoms_list.append(symptoms)
    disease_list.append(disease)

df = pd.DataFrame({'Symptoms': symptoms_list, 'Disease': disease_list})

# Преобразуем заболевания в числовой формат
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Инициализация токенизатора для симптомов
symptom_tokenizer = Tokenizer()
symptom_tokenizer.fit_on_texts(df['Symptoms'])
symptom_sequences = symptom_tokenizer.texts_to_sequences(df['Symptoms'])
symptom_padded = pad_sequences(symptom_sequences)

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(symptom_padded, df['Disease'], test_size=0.2, random_state=42)

# Создаем и обучаем нейронную сеть
model = Sequential([
    Embedding(input_dim=len(symptom_tokenizer.word_index) + 1, output_dim=10, input_length=symptom_padded.shape[1]),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(data), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Пример использования обученной модели
symptoms_input = input("Введите симптомы через запятую: ")
symptoms_array = np.array([symptoms_input])
symptom_sequences_input = symptom_tokenizer.texts_to_sequences(symptoms_array)
symptom_padded_input = pad_sequences(symptom_sequences_input, maxlen=symptom_padded.shape[1])
prediction = model.predict(symptom_padded_input)

# Выводим вероятности для каждого диагноза
all_diseases = label_encoder.inverse_transform(range(len(data)))

print("Вероятности для каждого диагноза:")
for disease, probability in zip(all_diseases, prediction[0]):
    print(f"{disease}: {probability * 100:.2f}%")

predicted_class = np.argmax(prediction[0])
predicted_disease = label_encoder.classes_[predicted_class]

print(f"\nПредсказанное заболевание: {predicted_disease}")

# Получаем фактический диагноз, который вы считаете правильным
correct_diagnosis = input("Введите правильный диагноз: ")

# Обработка неизвестных диагнозов
if correct_diagnosis not in label_encoder.classes_:
    print(f"Введенный диагноз '{correct_diagnosis}' неизвестен. Обработка неизвестных диагнозов...")
    # Можно реализовать обработку неизвестных диагнозов, не переобучив всю модель
else:
    # Обновляем обучающий набор данных
    new_row = {'Symptoms': symptoms_input, 'Disease': correct_diagnosis}
    df = df._append(new_row, ignore_index=True)
    # Преобразовываем заболевания в числовой формат
    df['Disease'] = label_encoder.fit_transform(df['Disease'])
    # Обновляем токенизатор для симптомов
    symptom_tokenizer.fit_on_texts(df['Symptoms'])
    symptom_sequences = symptom_tokenizer.texts_to_sequences(df['Symptoms'])
    symptom_padded = pad_sequences(symptom_sequences)
    # Разделяем данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(symptom_padded, df['Disease'], test_size=0.2, random_state=42)
    # Обновляем и переобучаем модель
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)