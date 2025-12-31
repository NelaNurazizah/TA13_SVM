import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen Tahun Baru 2026",
    page_icon="🎉",
    layout="wide"
)

# Judul dan Header
st.title("🎉 Analisis Sentimen: Tahun Baru 2026")
st.markdown("Aplikasi berbasis **Support Vector Machine (SVM)** untuk mendeteksi sentimen netizen.")
st.divider()

# ==========================================
# 2. FUNGSI-FUNGSI UTAMA (CACHED)
# ==========================================
# @st.cache_data membuat proses ini hanya jalan SEKALI saja saat web dibuka, biar ngebut.

@st.cache_resource
def load_and_train_model():
    # A. Load Data
    try:
        df = pd.read_csv('dataset_tahun_baru_2026.csv')
    except FileNotFoundError:
        st.error("File 'dataset_tahun_baru_2026.csv' tidak ditemukan! Pastikan file ada di folder yang sama.")
        return None, None, None, None, None

    # B. Preprocessing
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword = stopword_factory.create_stop_word_remover()

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = stopword.remove(text)
        text = stemmer.stem(text)
        return text
    
    df['clean_text'] = df['text'].apply(preprocess)
    
    # C. TF-IDF & Split
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label'] # 0: Negatif, 1: Positif
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # D. Train SVM
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    
    # E. Hitung Akurasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, acc, df, preprocess

# Load Model
model, vectorizer, accuracy, df, preprocess_func = load_and_train_model()

if model is not None:
    # Sidebar
    st.sidebar.header("Info Model")
    st.sidebar.success(f"Akurasi Model: **{accuracy*100:.2f}%**")
    st.sidebar.markdown("---")
    st.sidebar.write("**Dataset:** Tahun Baru 2026")
    st.sidebar.write(f"**Total Data:** {len(df)} baris")

    # ==========================================
    # 3. AREA PREDIKSI (DEMO)
    # ==========================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Uji Coba Prediksi")
        user_input = st.text_area("Masukkan komentar tentang Tahun Baru:", height=100, placeholder="Contoh: Macet banget parah, pemerintah gak becus!")
        
        if st.button("Analisis Sentimen", type="primary"):
            if user_input:
                # 1. Bersihkan input user
                clean_input = preprocess_func(user_input)
                # 2. Ubah ke angka (Vectorize)
                input_vec = vectorizer.transform([clean_input])
                # 3. Prediksi
                prediction = model.predict(input_vec)[0]
                
                # 4. Tampilkan Hasil
                st.markdown("### Hasil Analisis:")
                if prediction == 1:
                    st.success(f"😊 **SENTIMEN POSITIF**")
                    st.balloons()
                else:
                    st.error(f"😡 **SENTIMEN NEGATIF**")
            else:
                st.warning("Harap masukkan teks terlebih dahulu!")

    # ==========================================
    # 4. VISUALISASI (WORDCLOUD & DATA)
    # ==========================================
    with col2:
        st.subheader("📊 Visualisasi Data")
        tab1, tab2 = st.tabs(["WordCloud", "Lihat Data"])
        
        with tab1:
            st.write("Kata yang sering muncul:")
            # Generate Wordcloud (Simplified for Web)
            option = st.selectbox("Pilih Sentimen:", ["Positif", "Negatif"])
            
            label_code = 1 if option == "Positif" else 0
            text_wc = ' '.join(df[df['label'] == label_code]['clean_text'])
            colormap = 'Greens' if option == "Positif" else 'Reds'
            
            wc = WordCloud(width=400, height=300, background_color='white', colormap=colormap).generate(text_wc)
            
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
        with tab2:
            st.dataframe(df[['text', 'label']].head(10))
            st.caption("0: Negatif, 1: Positif")

else:
    st.stop()