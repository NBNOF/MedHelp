import json
import os
import random
from keras.models import load_model
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Размерность вложений=10, что мало. Увеличить чтобы повысить эффективность обработки более большого обьема данных
#Сложность модели недостаточно сложна для обработки и нахождения связи в больших обьемах. Добавить дополнительные слои или нейроны в существующие слои
#Главный недостаток - с увеличением сложности увеличивается риск переобучения. Техника регуляризации? dropout l1/l2 регулизация для предотвращение
#Недостаточная производительность - оптимизация гиперматаметров. Скорость обучения/кол-во epoch, размер пакета итд.


#Текущие недостатки:
#в случае обнаружения нового симптома или диагноза - модель придется переобучать

# Загрузка данных из JSON файла
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Создание DataFrame
symptoms_list = []
disease_list = []
for disease, symptoms in data.items():
    symptoms_list.append(symptoms)
    disease_list.append(disease)

df = pd.DataFrame({'Symptoms': symptoms_list, 'Disease': disease_list})

# Преобразование заболеваний в числовой формат
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Инициализация токенизатора для симптомов
symptom_tokenizer = Tokenizer()
symptom_tokenizer.fit_on_texts(df['Symptoms'])
symptom_sequences = symptom_tokenizer.texts_to_sequences(df['Symptoms'])
symptom_padded = pad_sequences(symptom_sequences)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(symptom_padded, df['Disease'], test_size=0.2, random_state=42)

# Создание и обучение нейронной сети
model = Sequential([
    Embedding(input_dim=len(symptom_tokenizer.word_index) + 1, output_dim=10, input_length=symptom_padded.shape[1]),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(data), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Проверка, существует ли уже обученная модель
if os.path.exists('my_model.h5'):
    model = load_model('my_model.h5')
else:
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)  # Добавление параметра verbose
    model.save('my_model.h5')
while True:
    # Пример использования обученной модели
    symptoms_input = input("Введите симптомы через запятую (или 'x' для выхода): ")
    if symptoms_input.lower() == 'x':
        model.save('my_model.h5')
        break
    symptoms_list = [symptom.strip() for symptom in symptoms_input.split(',')]
    symptoms_array = np.array([','.join(symptoms_list)])
    symptom_sequences_input = symptom_tokenizer.texts_to_sequences(symptoms_array)
    symptom_padded_input = pad_sequences(symptom_sequences_input, maxlen=symptom_padded.shape[1])
    prediction = model.predict(symptom_padded_input)

    # Вывод вероятностей для каждого диагноза
    all_diseases = label_encoder.inverse_transform(range(len(data)))
    print("Вероятности для каждого диагноза:")
    for disease, probability in zip(all_diseases, prediction[0]):
        print(f"{disease}: {probability * 100:.2f}%")

    # Получение предсказанного класса и заболевания
    predicted_class = np.argmax(prediction[0])
    predicted_disease = label_encoder.classes_[predicted_class]
    print(f"\nПредсказанное заболевание: {predicted_disease}")

    # Получение фактического диагноза от пользователя
    correct_diagnosis = input("Введите правильный диагноз (или 'x' для выхода): ")
    if correct_diagnosis.lower() == 'x':
        model.save('my_model.h5')
        break

    # Обработка неизвестных диагнозов
    if correct_diagnosis not in label_encoder.classes_:
        print(f"Введенный диагноз '{correct_diagnosis}' неизвестен. Обработка неизвестных диагнозов...")
        # Если отсутствует, добавляем новый диагноз в label_encoder
        label_encoder.classes_ = np.append(label_encoder.classes_, correct_diagnosis)

    # Обновление обучающего набора данных
    new_row = {'Symptoms': symptoms_input, 'Disease': correct_diagnosis}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Проверка и преобразование диагнозов в числовой формат
    df['Disease'] = df['Disease'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else x)

    # Преобразование заболеваний в числовой формат
    df['Disease'] = df['Disease'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else x)

    # Обновляем токенизатор для симптомов
    symptom_tokenizer.fit_on_texts(df['Symptoms'])
    symptom_sequences = symptom_tokenizer.texts_to_sequences(df['Symptoms'])
    symptom_padded = pad_sequences(symptom_sequences)

    # Разделяем данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(symptom_padded, df['Disease'], test_size=0.2, random_state=42)

    # Обучение модели
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
