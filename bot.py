import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Группы символов
char_groups = {
    "russian": "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    "english_espanol": "abcçdefghijklmnñopqrstuvwxyzABCÇDEFGHIJKLMNÑOPQRSTUVWXYZáéíóúÁÉÍÓÚ",
    "empty": "",
    "numbers": "0123456789",
    "symbols": "!@#$%^&*()_+-=[]{}|;:'\",.<>?/`~",
    "finance": "$£€¢₽₹¥₱₿",
    "math": "+-*/=<>≠≤≥",
    "science": "∞∑∏∫≡≠≈",
    "programmer": "<>{}[]()#@&|\\"
}

# Подготовка данных
texts = [
    " abcçdefghijklmnñopqrstuvwxyz ABCÇDEFGHIJKLMNÑOPQRSTUVWXYZ áéíóú ÁÉÍÓÚ абвгдеёжзийклмнопрстуфхцчшщъыьэюя АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ¡!@#№$£€¢₽₹¥₱₿%^&()_[]{}:;'\\|/<>¿?,.· 0123456789 +-*=≠≤≥∞∑∏∫≡≈ `~ºª",
    "",
    "Привет, как дела?",
    "Я хорошо, спасибо!",
    "Какой твой любимый цвет?",
    "Мне нравится синий.",
    "Что ты любишь делать в свободное время?",
    "Я люблю читать книги и смотреть фильмы.",
    "Кем ты работаешь?",
    "Я работаю программистом.",
    "Где ты живешь?",
    "Я живу в Москве.",
    "Какой у тебя хобби?",
    "Мое хобби - фотография.",
    "Какой твой любимый фильм?",
    "Мой любимый фильм - 'Интерстеллар'.",
    "Как ты проводишь выходные?",
    "Я люблю гулять парком.",
    "Что ты думаешь о погоде сегодня?",
    "Погода сегодня прекрасная и солнечная!",
    "У тебя есть домашние животные?",
    "Да, у меня есть собака и кошка.",
    "Какой твой любимый праздник?",
    "Мой любимый праздник - Новый год.",
    "Что ты планируешь на будущее?",
    "Я планирую путешествовать больше.",
    "Как ты расстаешься с друзьями?",
    "Я обычно говорю 'до встречи!'",
    "Как ты отмечаешь день рождения?",
    "Я праздную с друзьями и семьей.",
    "Hi, how are you?",
    "I'm fine, thank you!",
    "What's your favorite color?",
    "I like blue.",
    "What do you like to do in your free time?",
    "I like reading books and watching movies.",
    "What do you do for a living?",
    "I work as a programmer.",
    "Where do you live?",
    "I live in Moscow.",
    "What's your hobby?",
    "My hobby is photography.",
    "What's your favorite movie?",
    "My favorite movie is 'Interstellar'.",
    "How do you spend your weekends?",
    "I like walking in the park.",
    "What do you think about the weather today?",
    "The weather is beautiful and sunny today!",
    "Do you have pets?",
    "Yes, I have a dog and a cat.",
    "What's your favorite holiday?",
    "My favorite holiday is New Year.",
    "What are your plans for the future?",
    "I plan to travel more.",
    "How are you say goodbye to your friends?",
    "I usually say 'see you later!'",
    "How do you celebrate your birthday?",
    "I celebrate with friends and family.",
    "¿Hola, cómo estás?",
    "¡Estoy bien gracias!",
    "¿Cuál es tu color favorito?"
    "Me gusta el azul.",
    "¿Qué te gusta hacer en tu tiempo libre?"
    "Me encanta leer libros y ver películas.",
    "¿A qué te dedicas?",
    "Trabajo como programador.",
    "¿Dónde vive?",
    "Vivo en Moscú.",
    "¿Cuál es tu pasatiempo?"
    "Mi hobby es la fotografía.",
    "¿Cuál es tu película favorita?"
    "Mi película favorita es 'Interstellar'.",
    "¿Cómo pasas tu fin de semana?"
    "Me gusta caminar por el parque.",
    "¿Qué opinas del clima de hoy?"
    "¡El clima hoy es hermoso y soleado!",
    "¿Tienes mascotas?"
    "Sí, tengo un perro y un gato.",
    "¿Cuál es tu fiesta favorita?"
    "Mi fiesta favorita es el Año Nuevo.",
    "¿Qué estás planeando para el futuro?"
    "Planeo viajar más.",
    "¿Cómo se rompe con los amigos?"
    "Normalmente digo '¡hasta luego!'"
    "¿Cómo celebras tu cumpleaños?"
    "Lo celebro con amigos y familiares."
]

# Создаем словарь символов
chars = sorted(list(set(' '.join(texts))))  # Уникальные символы
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}

# Параметры
max_len = 20  # Максимальная длина последовательности
step = 1  # Шаг для создания последовательностей

# Создание последовательностей и меток
sequences = []
next_chars = []

for text in texts:
    for i in range(0, len(text) - max_len, step):
        sequences.append([char_to_idx[ch] for ch in text[i:i + max_len]])
        next_chars.append(char_to_idx[text[i + max_len]])

# Преобразуем в массивы NumPy
X = np.array(sequences)
y = np.array(next_chars)

# Преобразуем y в категориальный формат
y = keras.utils.to_categorical(y, num_classes=len(chars))

# Создание модели
model = keras.Sequential([
    layers.Embedding(input_dim=len(chars), output_dim=64, input_length=max_len),
    layers.LSTM(128),
    layers.Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, batch_size=32, epochs=20)

# Функция для генерации текста
def generate_text(seed_text, length=100):
    generated = seed_text
    for _ in range(length):
        # Преобразуем текст в последовательность индексов
        input_sequence = np.array([[char_to_idx.get(ch, len(char_to_idx)) for ch in generated[-max_len:]]])
        
        # Предсказание следующего символа
        preds = model.predict(input_sequence, verbose=0)
        next_index = np.argmax(preds)  # Индекс символа с максимальной вероятностью

        # Проверяем, что индекс находится в допустимом диапазоне
        if next_index >= len(chars):
            next_index = len(chars) - 1

        next_char = idx_to_char[next_index]

        generated += next_char
    
    return generated

while True:
    seed_text = str(input("Вы: "))
    try:
        texts.append(seed_text)
    except:
        continue
    
    # Пример добавления символа
    try:
        for char in seed_text:
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx)
    except:
        continue
    
    # Ответ Нейросети
    try:
        model.fit(X, y, batch_size=32, epochs=5)
    except:
        continue
    
    generated_text = generate_text(seed_text, length=100)

    # Вырезаем вопрос из ответа
    if generated_text.startswith(seed_text):
        generated_text = generated_text[len(seed_text):].strip()

    print("Сгенерированный текст:", generated_text)
