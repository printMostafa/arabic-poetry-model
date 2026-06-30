# arabic-poetry-model

# 📜 Arabic Poetry Generation using Character-Level LSTM

An artificial intelligence model built from scratch using Recurrent Neural Networks (RNN) with **LSTM** cells. The model was trained to autonomously learn Arabic grammar, poetic meters (بحور الشعر), and rhyming patterns directly at the character level, enabling it to improvise classical Arabic poetry.

---

## 🚀 Project Features
* **Massive Dataset:** Cleaned and trained on a massive poetic corpus containing nearly **98 million characters**.
* **Stable Training:** Integrated `Dropout` layers to prevent overfitting and ensure a smooth, steady decline in loss.
* **High-Quality Improvisation:** Capable of generating contextually coherent verses and capturing sophisticated classical vocabulary (e.g., المنايا, الترسل, يقري, الندامى).
* **Creativity Control:** Features an advanced generation function with a `Temperature` parameter to control the randomness and creativity of the model's output.

---

## 📊 Training Profile & Specifications
* **Architecture:** `LSTM (256 units)` -> `Dropout` -> `Dense (46 units)` -> `Softmax`
* **Optimizer:** `RMSprop` with a carefully tuned learning rate of `0.0005`.
* **Loss Function:** `categorical_crossentropy`
* **Vocabulary Size:** 46 unique characters.
* **Hardware & Duration:** Trained locally on a CPU for 50 full Epochs (taking over 18 hours of continuous computation). The loss successfully converged from `2.29` down to **`1.85`**, with a validation loss of `1.83`.

---

## 🛠️ Prerequisites & Installation

To run the script and generate poetry locally, install the required packages:
```bash
pip install tensorflow numpy pyarabic bidi arabic_reshaper

2. Tweaking Creativity via Temperature
temperature = 0.3: Safe and highly conservative generation. It strictly adheres to the most frequent structures found in the data.

temperature = 0.5: The sweet spot. Offers a perfect balance between proper grammar and creative improvisation (Recommended).

temperature = 0.8: Bold and highly creative. Produces unexpected, rich metaphors and diverse poetic imagery.

📝 Generation Sample
When fed the famous opening line from the Mu'allaqah of Imru' al-Qais, the model improvised the following extension:

Seed Text: قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ
Model Output: وَالشَّمْسُ فِي الْأَحْسَابِ مِنْ أَعْلَى الْبِلَادِ ... أَيَّا مَنْ يُرَاحُ بِالْمُسْتَحِيلِ الْمَسْرَى وَلَا أَنْبَاءَ مَا أَنْتَ مَنْ يَحْمُو
