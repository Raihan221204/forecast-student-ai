import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AI Student Forecaster",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# 2. LOAD ASSETS & DATA CLEANING
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan file .pkl ada di folder yang sama
    return joblib.load('final_model_xgb.pkl')

@st.cache_data
def load_data():
    # Load data CSV
    try:
        df = pd.read_csv('clean_scholarship_data_2023_2025.csv')
    except FileNotFoundError:
        st.error("File 'clean_scholarship_data_2023_2025.csv' tidak ditemukan.")
        return pd.DataFrame()

    # --- AUTO CLEANING (PEMBERSIHAN DATA OTOMATIS) ---
    # 1. Normalisasi Nama Kolom Tanggal
    if 'bulan' in df.columns:
        df['Month'] = df['bulan']
    
    # 2. Bersihkan Kolom Uang (Spending_Marketing) dari 'Rp' dan ','
    if 'Spending_Marketing' in df.columns:
        # Ubah ke string dulu untuk manipulasi teks
        df['Spending_Marketing'] = df['Spending_Marketing'].astype(str)
        # Hapus Rp dan Koma
        df['Spending_Marketing'] = df['Spending_Marketing'].str.replace('Rp', '', regex=False)
        df['Spending_Marketing'] = df['Spending_Marketing'].str.replace(',', '', regex=False)
        # Ubah ke Angka (Float)
        df['Spending_Marketing'] = pd.to_numeric(df['Spending_Marketing'], errors='coerce').fillna(0)

    # 3. Pastikan kolom angka lain bersih
    for col in ['student', 'Beasiswa']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Parsing Tanggal
    if 'Month' in df.columns:
        df['Date'] = pd.to_datetime(df['Month'])
    
    return df

# Eksekusi Load Data
try:
    model = load_model()
    df_history = load_data()
    
    if df_history.empty:
        st.stop()
    
    # --- PERBAIKAN LOGIKA LAG (REAL DATA) ---
    # Kita butuh data 2 bulan terakhir untuk Lag_1 dan Lag_2
    if len(df_history) >= 2:
        last_row = df_history.iloc[-1]       # Data Terakhir (Misal: Okt 2025)
        second_last_row = df_history.iloc[-2] # Data Sebelum Terakhir (Misal: Sep 2025)
        
        last_student = last_row['student']         # Ini jadi Lag_1
        prev_student = second_last_row['student']  # Ini jadi Lag_2 (REAL, BUKAN ESTIMASI)
    else:
        # Fallback kalau datanya cuma 1 baris (jarang terjadi)
        last_student = df_history.iloc[-1]['student']
        prev_student = last_student # Asumsi flat
    
    # Hitung Rata-rata historis
    avg_marketing = df_history['Spending_Marketing'].mean()
    avg_beasiswa = df_history['Beasiswa'].mean()
    
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: KONTROL PANEL
# ==========================================
st.sidebar.header("🎛️ Panel Kontrol")

# Pilihan Mode
mode = st.sidebar.radio(
    "Pilih Mode Prediksi:",
    ("🤖 Auto Pilot (Data Historis)", "🧪 Simulasi Manual")
)

st.sidebar.markdown("---")

# Input Tanggal
predict_date = st.sidebar.date_input("Target Bulan", datetime.now())

# Logika Input Berdasarkan Mode
if mode == "🧪 Simulasi Manual":
    st.sidebar.subheader("Input Skenario")
    marketing_input = st.sidebar.number_input("Budget Marketing (Rp)", value=500_000_000, step=10_000_000, format="%d")
    beasiswa_input = st.sidebar.number_input("Event Beasiswa", value=15, step=1)
    st.sidebar.info("Mode: Menggunakan angka input manual Anda.")
else:
    # Mode Otomatis (Pakai Rata-rata)
    marketing_input = avg_marketing
    beasiswa_input = avg_beasiswa
    st.sidebar.info(f"Mode: Menggunakan rata-rata historis.\n• Mkt: Rp {marketing_input:,.0f}\n• Event: {int(beasiswa_input)}")

run_predict = st.sidebar.button("🚀 Jalankan Prediksi", type="primary")

# ==========================================
# 4. ENGINE PREDIKSI (AI)
# ==========================================
predicted_student = 0

if run_predict:
    # Siapkan Data Input
    input_data = pd.DataFrame({
        'Spending_Marketing': [marketing_input],
        'Beasiswa': [beasiswa_input],
        'Month_Num': [predict_date.month],
        'Year': [predict_date.year],
        'Lag_1': [last_student],       
        'Lag_2': [prev_student]  # <--- SEKARANG SUDAH PAKAI DATA ASLI (Real September)
    })
    
    # Prediksi
    try:
        prediction = model.predict(input_data)
        predicted_student = int(prediction[0])
    except Exception as e:
        st.error(f"Gagal memprediksi: {e}")
# ==========================================
# 5. DASHBOARD UTAMA
# ==========================================
st.title("🎓 Intelligent Student Forecast & Planning")
st.markdown("Sistem prediksi berbasis AI untuk estimasi jumlah student dan perencanaan kapasitas tutor.")

tab1, tab2 = st.tabs(["📊 Dashboard & Kalkulator", "📂 Data Historis"])

with tab1:
    # --- SECTION A: HASIL FORECASTING ---
    st.subheader(f"1️⃣ Hasil Prediksi AI: {predict_date.strftime('%B %Y')}")
    
    if run_predict:
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Prediksi Student", f"{predicted_student} Siswa", f"{predicted_student - last_student} vs Bulan Lalu")
        c2.metric("💰 Budget Marketing", f"Rp {marketing_input:,.0f}")
        c3.metric("📅 Event Beasiswa", f"{int(beasiswa_input)} Event")
        
        # Grafik Mini Interaktif
        df_pred = pd.DataFrame({'Date': [pd.to_datetime(predict_date)], 'student': [predicted_student], 'Type': ['Forecast']})
        df_viz = df_history[['Date', 'student']].copy()
        df_viz['Type'] = 'Historical'
        
        # Gabung data history dan prediksi
        df_final_viz = pd.concat([df_viz, df_pred], ignore_index=True)
        
        fig = px.line(df_final_viz, x='Date', y='student', color='Type', markers=True, 
                      color_discrete_map={'Historical': 'blue', 'Forecast': 'red'},
                      title="Tren Historis & Posisi Forecasting")
        
        # Buat garis prediksi putus-putus
        fig.update_traces(patch={"line": {"dash": "dot"}}, selector={"legendgroup": "Forecast"}) 
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("👈 Silakan klik tombol 'Jalankan Prediksi' di sidebar untuk memulai.")

    st.markdown("---")

    # --- SECTION B: KALKULATOR BEBAN KERJA (MANDIRI) ---
    st.subheader("2️⃣ Kalkulator Kapasitas & Keadilan Tutor")
    st.caption("Gunakan bagian ini untuk menghitung kebutuhan SDM secara dinamis.")

    # Container Input Kalkulator
    with st.container(border=True):
        col_input, col_result = st.columns([1, 2])
        
        with col_input:
            # Nilai default diambil dari hasil prediksi (jika ada), kalau belum ada pakai data terakhir
            default_val = predicted_student if predicted_student > 0 else int(last_student)
            
            calc_student = st.number_input("Jumlah Student Aktif", value=default_val, help="Otomatis terisi hasil prediksi, tapi bisa diedit manual.")
            
            st.markdown("#### ⚙️ Parameter Kerja")
            # Slider Parameter
            avg_hours_student = st.slider("Jam per Student/Minggu", 0.5, 5.0, 1.5, 0.1)
            avg_hours_tutor = st.slider("Kapasitas Tutor/Minggu", 5.0, 40.0, 12.0, 1.0)
            
        with col_result:
            # --- RUMUS & LOGIC ---
            total_jam_needed = calc_student * avg_hours_student
            if avg_hours_tutor > 0:
                tutors_needed_float = total_jam_needed / avg_hours_tutor
                tutors_needed_round = int(tutors_needed_float) + 1 # Pembulatan ke atas
                
                # Rumus Keadilan (Fairness)
                max_student_per_tutor = int(avg_hours_tutor / avg_hours_student)
                real_load = calc_student / tutors_needed_round if tutors_needed_round > 0 else 0
            else:
                tutors_needed_round = 0
                max_student_per_tutor = 0
                real_load = 0
            
            # TAMPILAN HASIL
            st.markdown("### 📋 Analisis Kebutuhan SDM")
            
            m1, m2 = st.columns(2)
            m1.metric("Total Jam Mengajar", f"{total_jam_needed:,.1f} Jam")
            m2.metric("Tutor Dibutuhkan", f"{tutors_needed_round} Orang")
            
            st.divider()
            
            # Tampilan Insight Keadilan
            st.markdown("#### ⚖️ Analisis Beban Kerja (Fairness)")
            
            if tutors_needed_round > 0:
                st.write(f"Dengan **{tutors_needed_round} tutor**, rata-rata 1 tutor memegang **{real_load:.1f} siswa**.")
            
            # Rekomendasi Ideal
            if max_student_per_tutor > 0:
                st.success(f"💡 **Rekomendasi Ideal:** Agar adil & optimal, 1 Tutor sebaiknya maksimal memegang **{max_student_per_tutor} Siswa**.")
                st.caption(f"Rumus: Kapasitas Tutor ({avg_hours_tutor} jam) ÷ Kebutuhan Siswa ({avg_hours_student} jam)")

with tab2:
    st.markdown("### Data Historis (Cleaned)")
    st.dataframe(df_history, use_container_width=True)
