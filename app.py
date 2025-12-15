import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import datetime
import pandas as pd
import io
import database as db
from fpdf import FPDF
import hashlib
import pydicom
import os

# --- 1. SAYFA KONFİGÜRASYONU (Full Screen & Medical Title) ---
st.set_page_config(
    page_title="MedAI - Radyoloji Asistanı",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. KLİNİK CSS TASARIMI (Minimal & Modern) ---
st.markdown("""
<style>
    /* Genel Arkaplan */
    .stApp {
        background-color: #F8F9FA; /* Çok açık gri (Hastane beyazı) */
    }
    
    /* Sidebar Tasarımı */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Kart (Card) Yapısı */
    .medical-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #F0F0F0;
        margin-bottom: 20px;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Butonlar - Sidebar */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        background-color: white;
        color: #4A4A4A;
        transition: all 0.3s;
        text-align: left;
        padding-left: 15px;
    }
    .stButton button:hover {
        border-color: #007BFF;
        color: #007BFF;
        background-color: #F0F8FF;
    }
    
    /* Analiz CTA Butonu (Özel Stil) */
    .primary-cta button {
        background-color: #007BFF !important; /* Medikal Mavi */
        color: white !important;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3);
    }
    .primary-cta button:hover {
        background-color: #0056b3 !important;
    }
    
    /* Metrik Kutuları */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. TEMEL FONKSİYONLAR ---

# Veritabanı Başlatma
db.create_tables()
if not db.check_user_exists("admin"): db.add_user("admin", "12345")

# Session State
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'page' not in st.session_state: st.session_state['page'] = "Analiz"

# Model Yükleme
@st.cache_resource
def model_yukle():
    try: return tf.keras.models.load_model('yeni_coklu_model.keras')
    except: return None

# Yardımcılar
def tr_to_en(text):
    if not text: return ""
    degisim = {'ı':'i', 'İ':'I', 'ğ':'g', 'Ğ':'G', 'ü':'u', 'Ü':'U', 'ş':'s', 'Ş':'S', 'ö':'o', 'Ö':'O', 'ç':'c', 'Ç':'C'}
    for tr, en in degisim.items(): text = text.replace(tr, en)
    return text

def create_pdf(doctor, patient, pid, diagnosis, conf, note, date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(40, 10, 'TIBBI GORUNTULEME RAPORU'); pdf.ln(20)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=tr_to_en(f"Tarih: {date}"), ln=1)
    pdf.cell(200, 10, txt=tr_to_en(f"Doktor: {doctor}"), ln=1)
    pdf.cell(200, 10, txt=tr_to_en(f"Hasta: {patient} (ID: {pid})"), ln=1)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14); pdf.cell(200, 10, txt="AI ANALIZ SONUCU:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(255, 0, 0) if diagnosis != "Normal" else pdf.set_text_color(0, 128, 0)
    pdf.cell(200, 10, txt=tr_to_en(f"Teshis: {diagnosis} (Guven: %{conf:.2f})"), ln=1)
    pdf.set_text_color(0, 0, 0); pdf.ln(10)
    pdf.cell(200, 10, txt="Doktor Yorumu:", ln=1); pdf.multi_cell(0, 10, txt=tr_to_en(note if note else "Yorum girilmedi."))
    pdf.ln(20); pdf.set_font("Arial", 'I', 8); pdf.cell(200, 10, txt="Bu rapor AI desteklidir.", ln=1)
    return pdf.output(dest='S').encode('latin-1', 'ignore') 

def load_image_universal(uploaded_file):
    try:
        if uploaded_file.name.split('.')[-1].lower() == 'dcm':
            ds = pydicom.dcmread(uploaded_file); img = ds.pixel_array
            img = img - np.min(img); img = img / np.max(img); img = (img * 255).astype(np.uint8)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(img)
        return Image.open(uploaded_file).convert('RGB')
    except: return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    try:
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except: return np.zeros((224, 224))

def apply_filters(image, contrast, brightness, use_clahe, invert):
    img_array = np.array(image)
    img_array = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
    if len(img_array.shape) == 3: gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else: gray = img_array
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); gray = clahe.apply(gray)
        img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if invert: img_array = cv2.bitwise_not(img_array)
    return img_array

# --- 4. YENİ SIDEBAR YAPISI ---
def render_sidebar():
    with st.sidebar:
        # Profil Alanı (Avatar + Bilgi)
        prof = db.get_user_profile(st.session_state['username'])
        
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if prof and prof[3]:
            st.image(Image.open(io.BytesIO(prof[3])), width=100, use_column_width=False)
        else:
            # Avatar Placeholder (Klinik Hissi)
            st.markdown("""
                <div style="background-color: #E3F2FD; width: 80px; height: 80px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 30px;">👨‍⚕️</div>
            """, unsafe_allow_html=True)
        
        # İsim ve Rol
        doc_name = prof[0] if prof and prof[0] else st.session_state['username'].capitalize()
        doc_title = prof[1] if prof and prof[1] else "Radyoloji Uzmanı"
        
        st.markdown(f"<h3 style='margin-bottom: 0px; color: #007BFF;'>Dr. {doc_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #666; font-size: 14px; margin-top: -5px;'>{doc_title}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigasyon Butonları (İkonlu)
        if st.button("📊  Analiz & Rapor", use_container_width=True): st.session_state['page'] = "Analiz"; st.rerun()
        if st.button("📈  Dashboard (Panel)", use_container_width=True): st.session_state['page'] = "Dashboard"; st.rerun()
        if st.button("📂  Hasta Arşivi", use_container_width=True): st.session_state['page'] = "Kayitlar"; st.rerun()
        if st.button("👤  Profil Ayarları", use_container_width=True): st.session_state['page'] = "Profil"; st.rerun()
        
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Çıkış Yap", type="secondary", use_container_width=True): 
            st.session_state['logged_in'] = False
            st.rerun()

# --- 5. SAYFALAR ---

def analysis_page():
    # Sayfa Başlığı
    st.markdown("## 🩻 X-Ray Analiz İstasyonu")
    st.markdown("<p style='color:#666;'>Yapay zeka destekli görüntü işleme ve tanı asistanı</p>", unsafe_allow_html=True)
    st.markdown("---")

    # İki Sütunlu Yapı (Sol: Kontrol, Sağ: Görüntü)
    col_control, col_view = st.columns([1, 2.5], gap="large")
    
    # --- SOL PANEL: AYARLAR & YÜKLEME ---
    with col_control:
        # KART 1: Hasta Bilgileri
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Hasta Kaydı")
        h_ad = st.text_input("Hasta Adı Soyadı", placeholder="Örn: Ahmet Yılmaz")
        h_id = st.text_input("Protokol No", placeholder="123456")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # KART 2: Görüntü Ayarları
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Görüntü Filtreleri")
        
        con = st.slider("Kontrast Seviyesi", 0.5, 3.0, 1.0, help="Görüntüdeki zıtlığı artırır.")
        br = st.slider("Parlaklık", -100, 100, 0, help="Görüntü ışığını ayarlar.")
        
        c1, c2 = st.columns(2)
        with c1: clahe = st.checkbox("CLAHE (Netleştir)", help="Lokal kontrast iyileştirme")
        with c2: inv = st.checkbox("Negatif Mod", help="Kemik yapılarını belirginleştirir")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # KART 3: Dosya Yükleme (Medical Icon)
        st.markdown('<div class="medical-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("#### 📤 Görüntü Yükle")
        up = st.file_uploader("", type=['jpg','png','dcm'], label_visibility="collapsed")
        if up is None:
            st.markdown("📂 <br><small>DICOM, JPG, PNG</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- SAĞ PANEL: GÖRÜNTÜLEME & SONUÇ ---
    with col_view:
        if up:
            orig = load_image_universal(up)
            if orig:
                # Görüntü İşleme
                filt_arr = apply_filters(orig, con, br, clahe, inv)
                filt = Image.fromarray(filt_arr)
                
                # Görüntüleri Yan Yana Göster
                tab_g1, tab_g2 = st.tabs(["👁️ Önizleme & İşlem", "🔬 AI Detayları"])
                
                with tab_g1:
                    c_img1, c_img2 = st.columns(2)
                    c_img1.image(orig, caption="Orijinal Görüntü", use_container_width=True)
                    c_img2.image(filt, caption="Filtrelenmiş Görüntü", use_container_width=True)
                
                # CTA Butonu (Ortalanmış ve Büyük)
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2:
                    st.markdown('<div class="primary-cta">', unsafe_allow_html=True)
                    analyze = st.button("YAPAY ZEKA İLE ANALİZ ET ⚡", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Analiz Sonuçları
                if analyze:
                    if not h_ad: 
                        st.warning("Lütfen hasta adı giriniz.")
                    else:
                        with st.spinner("Görüntü işleniyor ve nöral ağ taranıyor..."):
                            model = model_yukle()
                            if model:
                                img_arr = np.array(orig); img_rez = cv2.resize(img_arr, (224,224)); img_fin = np.expand_dims(img_rez/255.0, axis=0)
                                preds = model.predict(img_fin)[0]
                                
                                classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                                idx = np.argmax(preds); res = classes[idx]; conf = preds[idx]*100
                                
                                # Sonuç Kartı
                                st.markdown("---")
                                st.markdown("### 📊 Analiz Raporu")
                                
                                c_res1, c_res2 = st.columns([1, 1])
                                
                                with c_res1:
                                    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
                                    if res == "Normal":
                                        st.success(f"✅ TESPİT: {res}")
                                        st.markdown(f"**Güven Skoru:** %{conf:.2f}")
                                    else:
                                        st.error(f"⚠️ BULGU: {res}")
                                        st.markdown(f"**Güven Skoru:** %{conf:.2f}")
                                        st.caption("AI, görüntüde patolojik bulgular saptadı.")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # PDF Butonu
                                    note = "Otomatik AI Analizi."
                                    db.add_record(st.session_state['username'], h_ad, h_id, res, float(conf), datetime.datetime.now().strftime("%Y-%m-%d"), note, "Onay")
                                    pdf_data = create_pdf(st.session_state['username'], h_ad, h_id, res, conf, note, datetime.datetime.now().strftime("%Y-%m-%d"))
                                    st.download_button("📄 RESMİ RAPORU İNDİR (PDF)", data=pdf_data, file_name=f"rapor_{h_id}.pdf", mime="application/pdf", use_container_width=True)

                                with c_res2:
                                    # Grafik
                                    chart_data = pd.DataFrame({"Durum": classes, "Olasılık": preds})
                                    st.bar_chart(chart_data.set_index("Durum"), color="#007BFF")
                                
                                # Heatmap (GradCAM)
                                if res != "Normal":
                                    with tab_g2:
                                        st.info("Hastalık Odak Haritası (Grad-CAM)")
                                        hm_img = np.clip(cv2.resize(cv2.cvtColor(cv2.applyColorMap(np.uint8(255*make_gradcam_heatmap(img_fin, model)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), (224,224))*0.4+img_rez,0,255).astype('uint8')
                                        st.image(hm_img, caption="AI Dikkat Alanı", width=300)
        else:
            # Boş Durum (Empty State) - Sağ taraf boş kalmasın diye
            st.info("👈 Analize başlamak için sol panelden bir röntgen görüntüsü yükleyin.")
            st.markdown("""
                <div style='text-align: center; color: #ccc; padding: 50px;'>
                    <h1>🩻</h1>
                    <p>Görüntü Bekleniyor...</p>
                </div>
            """, unsafe_allow_html=True)

def dashboard_page():
    st.markdown("## 📈 Klinik İstatistikler")
    data = db.get_all_stats()
    if data:
        df = pd.DataFrame(data, columns=['Teşhis', 'Durum'])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Hasta", len(df))
        c2.metric("COVID Vakası", len(df[df['Teşhis']=="COVID"]), delta_color="inverse")
        c3.metric("Normal", len(df[df['Teşhis']=="Normal"]))
        c4.metric("Onaylanan", len(df[df['Durum']!='Bekliyor']))
        
        st.markdown("### Hastalık Dağılımı")
        st.bar_chart(df['Teşhis'].value_counts(), color="#007BFF")
    else:
        st.info("Henüz veri bulunmuyor.")

def records_page():
    st.markdown("## 📂 Hasta Arşivi")
    recs = db.get_records_by_doctor(st.session_state['username'])
    if recs:
        df = pd.DataFrame(recs, columns=['ID','Dr','Hasta','Protokol','Teşhis','Skor','Tarih','Not','Durum'])
        st.dataframe(df[['Hasta','Protokol','Teşhis','Skor','Tarih','Durum']], use_container_width=True)
    else:
        st.info("Kayıt bulunamadı.")

def profile_page():
    st.markdown("## 👤 Profil Ayarları")
    u = st.session_state['username']
    data = db.get_user_profile(u)
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown('<div class="medical-card" style="text-align:center;">', unsafe_allow_html=True)
        if data and data[3]: st.image(Image.open(io.BytesIO(data[3])), width=150)
        else: st.markdown("<h1>👨‍⚕️</h1>", unsafe_allow_html=True)
        st.caption("Profil Fotoğrafı")
        new_pic = st.file_uploader("Değiştir", type=['png', 'jpg'])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        name = st.text_input("Ad Soyad", value=data[0] if data and data[0] else "")
        spec = st.text_input("Uzmanlık / Unvan", value=data[1] if data and data[1] else "")
        bio = st.text_area("Hakkımda", value=data[2] if data and data[2] else "")
        
        if st.button("💾 Profili Güncelle", type="primary"):
            blob = new_pic.getvalue() if new_pic else (data[3] if data else None)
            db.update_user_profile(u, name, spec, bio, blob)
            st.success("Bilgiler güncellendi!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def login_page():
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="medical-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("<h1>🩺 MedAI</h1>", unsafe_allow_html=True)
        st.markdown("<p>Güvenli Giriş Portalı</p>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
        with tab1:
            u = st.text_input("Kullanıcı Adı", key="l_u")
            p = st.text_input("Şifre", type="password", key="l_p")
            if st.button("Sisteme Gir", type="primary", use_container_width=True):
                if db.login_user(u, p):
                    st.session_state['logged_in'] = True; st.session_state['username'] = u; st.session_state['page'] = "Analiz"; st.rerun()
                else: st.error("Hatalı bilgiler.")
        
        with tab2:
            nu = st.text_input("Yeni Kullanıcı", key="r_u")
            np1 = st.text_input("Şifre Belirle", type="password", key="r_p1")
            np2 = st.text_input("Şifre Tekrar", type="password", key="r_p2")
            if st.button("Hesap Oluştur", use_container_width=True):
                if np1==np2 and nu:
                    if not db.check_user_exists(nu): db.add_user(nu, np1); st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
                    else: st.error("Kullanıcı adı dolu.")
                else: st.error("Şifreler uyuşmuyor.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- 6. ANA UYGULAMA AKIŞI ---
if st.session_state['logged_in']:
    render_sidebar() # Yeni Sidebarı Çağır
    
    if st.session_state['page'] == "Dashboard": dashboard_page()
    elif st.session_state['page'] == "Analiz": analysis_page()
    elif st.session_state['page'] == "Kayitlar": records_page()
    elif st.session_state['page'] == "Profil": profile_page()
else:
    login_page()