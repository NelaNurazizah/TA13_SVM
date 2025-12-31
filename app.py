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
from sklearn.metrics import accuracy_score
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
# 2. FUNGSI UTAMA (CACHE)
# ==========================================
@st.cache_resource
def load_and_train_model():
    # A. Load Data
    try:
        df = pd.read_csv('dataset_tahun_baru_2026.csv')
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan!")
        return None, None, None, None, None, 0, 0

    # Mapping Label
    df['Sentiment_Label'] = df['label'].map({0: 'Negatif', 1: 'Positif'})

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
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # D. Train SVM
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    
    # E. Akurasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # PERBAIKAN DI SINI: Gunakan .shape[0] alih-alih len()
    return model, vectorizer, acc, df, preprocess, X_train.shape[0], X_test.shape[0]

# Load Model
model, vectorizer, accuracy, df, preprocess_func, n_train, n_test = load_and_train_model()

if model is not None:
    # Sidebar
    st.sidebar.header("Info Model")
    st.sidebar.success(f"Akurasi Model: **{accuracy*100:.2f}%**")
    st.sidebar.markdown("---")
    st.sidebar.info(f"Total Data: {len(df)}")
    st.sidebar.write(f"Data Latih: {n_train}")
    st.sidebar.write(f"Data Uji: {n_test}")

    # ==========================================
    # 3. AREA UTAMA
    # ==========================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Uji Coba Prediksi")
        user_input = st.text_area("Masukkan komentar:", height=100, placeholder="Contoh: Macet banget parah, pemerintah gak becus!")
        
        if st.button("Analisis Sentimen", type="primary"):
            if user_input:
                clean_input = preprocess_func(user_input)
                input_vec = vectorizer.transform([clean_input])
                prediction = model.predict(input_vec)[0]
                
                st.markdown("### Hasil Analisis:")
                if prediction == 1:
                    st.success(f"😊 **SENTIMEN POSITIF**")
                    st.balloons()
                else:
                    st.error(f"😡 **SENTIMEN NEGATIF**")
            else:
                st.warning("Harap masukkan teks!")

    # ==========================================
    # 4. VISUALISASI (TAB BARU)
    # ==========================================
    with col2:
        st.subheader("📊 Visualisasi & Data")
        tab1, tab2, tab3 = st.tabs(["Statistik", "WordCloud", "Data Mentah"])
        
        # TAB 1: GRAFIK BATANG
        with tab1:
            st.write("**1. Distribusi Sentimen**")
            sentiment_counts = df['Sentiment_Label'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel', ax=ax1)
            ax1.set_ylabel("Jumlah")
            st.pyplot(fig1)

            st.write("**2. Split Data**")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            # Pastikan n_train dan n_test adalah integer
            ax2.bar(['Latih', 'Uji'], [n_train, n_test], color=['green', 'red'])
            st.pyplot(fig2)

        # TAB 2: WORDCLOUD
        with tab2:
            st.write("**Kata Paling Sering Muncul:**")
            option = st.selectbox("Pilih Kelas:", ["Positif", "Negatif"])
            label_code = 1 if option == "Positif" else 0
            
            # Filter teks berdasarkan label
            text_data = df[df['label'] == label_code]['clean_text']
            
            # Cek jika data kosong untuk menghindari error WordCloud kosong
            if len(text_data) > 0:
                text_wc = ' '.join(text_data)
                color_wc = 'Greens' if option == "Positif" else 'Reds'
                
                wc = WordCloud(width=400, height=300, background_color='white', colormap=color_wc).generate(text_wc)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.warning("Tidak ada kata untuk ditampilkan.")

        # TAB 3: DATA MENTAH
        with tab3:
            st.dataframe(df[['text', 'Sentiment_Label']], height=300)

else:
    st.stop()