import streamlit as st
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Sayfa ayarları
st.set_page_config(page_title="🌤️ Hava Durumu Tahmini", layout="wide", initial_sidebar_state="expanded")

# CSS stillendirmesi
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .title-main {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.markdown('<div class="title-main">🌤️ Hava Durumu Tahmini Sistemi</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    model_choice = st.radio("Model Seçin:", ["Joblib", "Pickle"])
    tahmin_gunu = st.slider("Tahmin Periyodu (Gün):", 7, 90, 30)
    
    st.markdown("---")
    st.markdown("### 📊 Model Bilgisi")
    st.info(f"✓ Seçilen Model: {model_choice}\n✓ Tahmin Süresi: {tahmin_gunu} gün")

# Model yükleme
@st.cache_resource
def load_model(model_type):
    if model_type == "Joblib":
        return joblib.load('model.joblib')
    else:
        return pickle.load(open('model.pkl', 'rb'))

try:
    model = load_model(model_choice)
    st.success("✓ Model başarıyla yüklendi")
except:
    st.error("❌ Model dosyaları bulunamadı. Lütfen model.pkl ve model.joblib dosyalarını kontrol edin.")
    st.stop()

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Hava Durumu Tahmini")
    
    # Tahmin yap
    future = model.make_future_dataframe(periods=tahmin_gunu)
    forecast = model.predict(future)
    
    # Son gün verisi
    last_forecast = forecast.iloc[-1]
    
    # Metrikleri göster
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="🌡️ Tahmini Sıcaklık",
            value=f"{last_forecast['yhat']:.1f}°C",
            delta=f"±{(last_forecast['yhat_upper'] - last_forecast['yhat_lower'])/2:.1f}°C"
        )
    
    with metric_col2:
        st.metric(
            label="📅 Tahmin Tarihi",
            value=last_forecast['ds'].strftime('%d/%m/%Y')
        )
    
    with metric_col3:
        st.metric(
            label="📊 Tahmin Gün Sayısı",
            value=f"{tahmin_gunu} gün"
        )

with col2:
    st.markdown("### 📋 İstatistikler")
    forecast_tail = forecast.tail(tahmin_gunu)
    
    st.info(f"""
    **Ortalama Sıcaklık:** {forecast_tail['yhat'].mean():.2f}°C
    
    **Min Sıcaklık:** {forecast_tail['yhat'].min():.2f}°C
    
    **Max Sıcaklık:** {forecast_tail['yhat'].max():.2f}°C
    """)

# Grafik 1: Zaman Serisi
st.markdown("---")
st.markdown("### 📊 Grafikler")

tab1, tab2, tab3 = st.tabs(["📈 Zaman Serisi", "📉 Günlük Tahmin", "🔍 Detaylı Analiz"])

with tab1:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    # Son 365 gün gerçek + tahmin
    historical = forecast[forecast['ds'] <= forecast['ds'].max() - timedelta(days=tahmin_gunu)]
    
    ax.plot(historical['ds'], historical['yhat'], 'b-', linewidth=2, label='Geçmiş Veriler', alpha=0.7)
    ax.plot(forecast_tail['ds'], forecast_tail['yhat'], 'r--', linewidth=2.5, label='Tahmin', alpha=0.9)
    ax.fill_between(forecast_tail['ds'], forecast_tail['yhat_lower'], 
                     forecast_tail['yhat_upper'], alpha=0.2, color='red', label='Güven Aralığı')
    
    ax.set_xlabel('Tarih', fontweight='bold', fontsize=11)
    ax.set_ylabel('Sıcaklık (°C)', fontweight='bold', fontsize=11)
    ax.set_title('Sıcaklık Tahmini - Zaman Serisi', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

with tab2:
    # Son 30 günlük tahmin
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    recent_forecast = forecast_tail.tail(30) if len(forecast_tail) >= 30 else forecast_tail
    colors = ['#2ECC71' if x > recent_forecast['yhat'].mean() else '#E74C3C' 
              for x in recent_forecast['yhat']]
    
    ax.bar(range(len(recent_forecast)), recent_forecast['yhat'], color=colors, 
           edgecolor='#2C3E50', alpha=0.7)
    ax.axhline(y=recent_forecast['yhat'].mean(), color='#3498DB', linestyle='--', 
               linewidth=2.5, label=f'Ortalama: {recent_forecast["yhat"].mean():.1f}°C')
    
    ax.set_xlabel('Gün', fontweight='bold', fontsize=11)
    ax.set_ylabel('Sıcaklık (°C)', fontweight='bold', fontsize=11)
    ax.set_title('Günlük Sıcaklık Tahmini', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    st.pyplot(fig)

with tab3:
    # Min-Max aralık
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    ax.plot(forecast_tail['ds'], forecast_tail['yhat'], 'o-', linewidth=2.5, 
            markersize=6, color='#3498DB', label='Tahmin', alpha=0.8)
    ax.fill_between(forecast_tail['ds'], forecast_tail['yhat_lower'], 
                     forecast_tail['yhat_upper'], alpha=0.3, color='#3498DB', 
                     label='95% Güven Aralığı')
    
    ax.set_xlabel('Tarih', fontweight='bold', fontsize=11)
    ax.set_ylabel('Sıcaklık (°C)', fontweight='bold', fontsize=11)
    ax.set_title('Detaylı Tahmin - Min/Max Aralığı', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

# Veri Tablosu
st.markdown("---")
st.markdown("### 📋 Tahmin Verisi")

display_df = forecast_tail[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
display_df.columns = ['Tarih', 'Tahmin', 'Alt Sınır', 'Üst Sınır']
display_df['Tarih'] = display_df['Tarih'].dt.strftime('%d/%m/%Y')
display_df['Tahmin'] = display_df['Tahmin'].round(2)
display_df['Alt Sınır'] = display_df['Alt Sınır'].round(2)
display_df['Üst Sınır'] = display_df['Üst Sınır'].round(2)

st.dataframe(display_df, use_container_width=True)

# Download butonu
csv = display_df.to_csv(index=False)
st.download_button(
    label="📥 Tahmin Verilerini İndir (CSV)",
    data=csv,
    file_name=f"tahmin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    🔮 Prophet Zaman Serisi Tahmini Sistemi | 
    <span style='color: #667eea;'>tugcesi/Wheather-Forecasting</span>
    </div>
""", unsafe_allow_html=True)