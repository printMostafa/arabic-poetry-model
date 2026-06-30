import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from tensorflow.keras.callbacks import EarlyStopping
import pickle


filePath = 'APCD_plus_porse_all.csv'

text = open(filePath,'rb').read().decode(encoding='utf-8')

lines = text.split('\n')

clean_lines = []
for line in lines:
    if ',' in line:
        poetry_line = line.split(',')[0]
        clean_lines.append(poetry_line)
    else:
        clean_lines.append(line)
        
text = "\n".join(clean_lines)

print(f"طول النص بعد التنظيف: {len(text)} حرف")

# print("\n--- أول 5 سطور من الداتا ---")
# for line in text.split('\n')[:5]:
#     # reshape بتدمج الحروف صح، و get_display بتعكس الاتجاه عشان التيرمينال تقراه معدول
#     reshaped_text = reshape(line)
#     bidi_text = get_display(reshaped_text)
#     print(bidi_text)

text = text[:30000000]

characters = sorted(list(set(text)))
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}

print(f"عدد الحروف الفريدة في القاموس: {len(characters)}")

sentences = []
next_chars = []
SEQUENCE_LENGTH = 40
STEP = 5

for i in range(0, len(text) - SEQUENCE_LENGTH, STEP):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(characters)), dtype='bool')
y = np.zeros((len(sentences), len(characters)), dtype='bool')

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
for i, char in enumerate(next_chars):
    y[i, char_to_index[char]] = 1
    
# model = Sequential([
#     Input(shape=(SEQUENCE_LENGTH, len(characters))),
#     LSTM(256),
#     Dropout(0.2),
#     Dense(len(characters), activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0005))
# model.summary()
# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )
# print(tf.config.list_physical_devices('GPU'))
# model.fit(
#     x,
#     y,
#     batch_size=128,
#     epochs=50,
#     validation_split=0.1,
#     callbacks=[early_stop]
# )
# model.save("arabic_poetry_model.keras")

model = tf.keras.models.load_model("arabic_poetry_model.keras")
with open("char_dict.pkl", "wb") as f:
    pickle.dump(
        (characters, char_to_index, index_to_char),
        f
    )
def sample(preds, temperature=0.4):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_poetry(model, seed_text, characters, char_to_index, index_to_char, SEQUENCE_LENGTH=40, length=200, temperature=0.4):
    generated_text = seed_text
    
    print(f"\n--- جملة البداية: {get_display(reshape(seed_text))} ---")
    print("--- الموديل يرتجل الآن حرفاً بحرف: ---\n")
    
    for _ in range(length):
        # الميزة العبقرية اللي أنت عملتها: أخذ آخر 40 حرف دايماً
        current_batch = generated_text[-SEQUENCE_LENGTH:]
        
        x_pred = np.zeros((1, SEQUENCE_LENGTH, len(characters)), dtype='float32')
        if len(seed_text) < SEQUENCE_LENGTH:
            seed_text = " " * (SEQUENCE_LENGTH - len(seed_text)) + seed_text
        for t, char in enumerate(current_batch):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated_text += next_char
        
        # اطبع الحرف فوراً بشكل معدول ومن غير سطر جديد عشان يشبك في اللي قبله
        print(get_display(reshape(next_char)), end='', flush=True)
        
    print("\n\n-------------------------")
        # بعد ما الـ loop بتاعة التوليد تخلص تماماً وتجمع الـ generated_text:
    print("\n\n================ القصيدة الارتجالية ================\n")
    # بنقسم النص لسطور، ونعكس كل سطر بالـ reshape والـ display عشان الـ Terminal تفهمه صح
    for line in generated_text.split('\n'):
        print(get_display(reshape(line)))
    print("\n====================================================")
    return generated_text

seed = "قفا نبك من ذكرى حبيب ومنزل"

generate_poetry(
    model,
    seed,
    characters,
    char_to_index,
    index_to_char,
    SEQUENCE_LENGTH=40,
    length=300,
    temperature=0.4
)
