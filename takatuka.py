python

import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Verileri yükle
with open("soru_cevap.txt", "r") as f:
    sorular = []
    cevaplar = []
    for line in f:
        soru, cevap = line.strip().split(":")
        sorular.append(soru)
        cevaplar.append(cevap)

# Doğal dil işleme için uygun hale getir
def preprocess(text):
    # Büyük/küçük harf dönüşümü
    text = text.lower()
    # Noktalama işaretlerinin kaldırılması
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Stop words'lerin çıkarılması
    stopwords = nltk.corpus.stopwords.words("english")
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    # Token'ların birleştirilmesi
    text = " ".join(tokens)
# Verileri doğal dil işleme için uygun hale getir
corpus = []
for soru in sorular:
    preprocessed_soru = preprocess(soru)
    corpus.append(preprocessed_soru)

# CountVectorizer kullanarak vektörleştirme yap
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Multinomial Naive Bayes sınıflandırıcısı kullanarak modeli eğit
model = MultinomialNB()
model.fit(X, cevaplar)

# Kullanıcıdan soru al ve modele girdi olarak ver
soru = input("Soru: ")
soru = preprocess(soru)
soru = vectorizer.transform([soru])

# Modelden cevap al
cevap = model.predict(soru)

# Cevap anahtarından doğru cevabı al
index = np.where(cevaplar == cevap)[0][0]
dogru_cevap = cevaplar[index]

# Yanıtın doğruluğunu hesapla
if cevap == dogru_cevap:
    print("Doğru!")
else:
    print("Yanlış. Doğru cevap: " + dogru_cevap
