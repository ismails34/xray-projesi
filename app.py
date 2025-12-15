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

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="X-Ray Vision Pro", layout="wide", initial_sidebar_state="expanded")

# --- TASARIM (KREM & LATTE - GERİ GELDİ!) ---
st.markdown("""
<style>
    .stApp { background-color: #FDFBF7 !important; }
    .login-header { color: #5D4037; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 10px; }
    .login-sub { color: #8D6E63; text-align: center; margin-bottom: 30px; }
    div[data-testid="metric-container"] { background-color: #FFFFFF; border: 1px solid #F0E6D2; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .block-container { background-color: #FFFFFF; padding: 2rem; border-radius: 15px; border: 1px solid #F0E6D2; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .profile-pic { border-radius: 12px; width: 100%; max-width: 250px; border: 3px solid #D4A373; display: block; margin: 0 auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    section[data-testid="stSidebar"] { background-color: #F5F5DC !important; border-right: 1px solid #E6E6DA; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p { color: #4A4A4A !important; }
    section[data-testid="stSidebar"] h3 { color: #5D4037 !important; }
    h1, h2, h3, h4, label, div, span, p { color: #2C3E50; }
    .stButton>button { background-color: #D4A373 !important; color: white !important; border: none; border-radius: 8px; height: 48px; font-weight: bold; font-size: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stButton>button:hover { background-color: #BC8A5F !important; }
</style>
""", unsafe_allow_html=True)

# --- DB & GÜVENLİK ---
db.create_tables()
if not db.check_user_exists("admin"): db.add_user("admin", "12345")
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'page' not in st.session_state: st.session_state['page'] = "Analiz"

# --- TÜRKÇE KARAKTER DÜZELTİCİ (PDF HATASI ÇÖZÜMÜ) ---
def tr_to_en(text):
    if not text: return ""
    degisim = {'ı':'i', 'İ':'I', 'ğ':'g', 'Ğ':'G', 'ü':'u', 'Ü':'U', 'ş':'s', 'Ş':'S', 'ö':'o', 'Ö':'O', 'ç':'c', 'Ç':'C'}
    for tr, en in degisim.items():
        text = text.replace(tr, en)
    return text

# --- PDF OLUŞTURUCU ---
def create_pdf(doctor, patient, pid, diagnosis, conf, note, date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, 'TIBBI GORUNTULEME RAPORU')
    pdf.ln(20)
    
    # Türkçe karakterleri temizleyerek yaz
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=tr_to_en(f"Tarih: {date}"), ln=1)
    pdf.cell(200, 10, txt=tr_to_en(f"Doktor: {doctor}"), ln=1)
    pdf.cell(200, 10, txt=tr_to_en(f"Hasta: {patient} (ID: {pid})"), ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="AI ANALIZ SONUCU:", ln=1)
    pdf.set_font("Arial", size=12)
    
    if diagnosis == "Normal": pdf.set_text_color(0, 128, 0)
    else: pdf.set_text_color(255, 0, 0)
        
    pdf.cell(200, 10, txt=tr_to_en(f"Teshis: {diagnosis} (Guven: %{conf:.2f})"), ln=1)
    
    pdf.set_text_color(0, 0, 0); pdf.ln(10)
    pdf.cell(200, 10, txt="Doktor Yorumu:", ln=1)
    pdf.multi_cell(0, 10, txt=tr_to_en(note if note else "Yorum girilmedi."))
    
    pdf.ln(20); pdf.set_font("Arial", 'I', 8)
    pdf.cell(200, 10, txt="Bu rapor AI desteklidir. Kesin teshis degildir.", ln=1)
    
    # Latin-1 hatasını engellemek için 'ignore' kullanıyoruz
    return pdf.output(dest='S').encode('latin-1', 'ignore') 

# --- MODEL YÜKLE ---
@st.cache_resource
def model_yukle():
    try: return tf.keras.models.load_model('yeni_coklu_model.keras')
    except: return None

# --- GÖRÜNTÜ İŞLEME ---
def load_image_universal(uploaded_file):
    try:
        if uploaded_file.name.split('.')[-1].lower() == 'dcm':
            ds = pydicom.dcmread(uploaded_file)
            img = ds.pixel_array
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if invert: img_array = cv2.bitwise_not(img_array)
    return img_array

# --- EKRANLAR ---
def login_page():
    col_space1, col_main, col_space2 = st.columns([1, 2, 1])
    with col_main:
        st.markdown('<div class="login-header">Hastane Yönetim Sistemi</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">4-Sınıflı Hassas Radyoloji Modülü</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["GİRİŞ YAP", "KAYIT OL"])
        with tab1:
            st.write("")
            u = st.text_input("Kullanıcı Adı", key="log_u")
            p = st.text_input("Şifre", type="password", key="log_p")
            if st.button("SİSTEME GİRİŞ YAP", use_container_width=True, type="primary"):
                if db.login_user(u, p):
                    st.session_state['logged_in'] = True; st.session_state['username'] = u; st.session_state['page'] = "Analiz"; st.rerun()
                else: st.error("Hatalı bilgi!")
        with tab2:
            st.write("")
            nu = st.text_input("Kullanıcı Adı", key="reg_u")
            np1 = st.text_input("Şifre", type="password", key="reg_p1")
            np2 = st.text_input("Şifre Tekrar", type="password", key="reg_p2")
            if st.button("HESAP OLUŞTUR", use_container_width=True):
                if np1 == np2 and nu:
                    if not db.check_user_exists(nu): db.add_user(nu, np1); st.success("Hesap oluşturuldu!")
                    else: st.error("Kullanıcı adı dolu.")
                else: st.warning("Şifreler uyuşmuyor.")

def analysis_page():
    st.title("Gelişmiş Radyoloji İstasyonu")
    col_tools, col_main = st.columns([1, 2.5])
    
    with col_tools:
        st.markdown("### 🎛️ Görüntü Ayarları")
        st.info("4 Sınıf: Covid, Normal, Pnömoni, Opaklık")
        con = st.slider("Kontrast", 0.5, 3.0, 1.0)
        br = st.slider("Parlaklık", -100, 100, 0)
        clahe = st.checkbox("CLAHE Filtresi")
        inv = st.checkbox("Negatif Mod")
        st.markdown("---")
        h_ad = st.text_input("Hasta Adı")
        h_id = st.text_input("Protokol No")
        up = st.file_uploader("Dosya Yükle", type=['jpg','png','dcm','webp'])
        
    with col_main:
        if up:
            orig = load_image_universal(up)
            if orig:
                filt_arr = apply_filters(orig, con, br, clahe, inv)
                filt = Image.fromarray(filt_arr)
                col1, col2 = st.columns(2)
                col1.image(orig, caption="Orijinal", use_container_width=True)
                col2.image(filt, caption="İşlenmiş", use_container_width=True)
                
                if st.button("YAPAY ZEKA ANALİZİNİ BAŞLAT", type="primary", use_container_width=True):
                    if not h_ad: st.warning("Hasta adı giriniz."); return
                    
                    with st.spinner("Çoklu Hastalık Taraması Yapılıyor..."):
                        model = model_yukle()
                        if model:
                            img_arr = np.array(orig); img_rez = cv2.resize(img_arr, (224,224)); img_fin = np.expand_dims(img_rez/255.0, axis=0)
                            preds = model.predict(img_fin)[0]
                            
                            # 4 SINIF - ALFABETİK SIRA: ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                            classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                            idx = np.argmax(preds); res = classes[idx]; conf = preds[idx]*100
                            
                            cr, cg = st.columns([1,1])
                            with cr:
                                if res == "Normal": st.success(f"✅ {res} (%{conf:.2f})")
                                else: st.error(f"⚠️ {res} (%{conf:.2f})"); st.image(np.clip(cv2.resize(cv2.cvtColor(cv2.applyColorMap(np.uint8(255*make_gradcam_heatmap(img_fin, model)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), (224,224))*0.4+img_rez,0,255).astype('uint8'), caption="Odak", width=250)
                            with cg: st.bar_chart(pd.DataFrame({"Sınıf": classes, "Olasılık": preds}).set_index("Sınıf"))
                            
                            note = "Otomatik analiz."
                            db.add_record(st.session_state['username'], h_ad, h_id, res, float(conf), datetime.datetime.now().strftime("%Y-%m-%d"), note, "Onay")
                            pdf_data = create_pdf(st.session_state['username'], h_ad, h_id, res, conf, note, datetime.datetime.now().strftime("%Y-%m-%d"))
                            st.download_button("RAPOR İNDİR (PDF)", data=pdf_data, file_name=f"rapor_{h_id}.pdf", mime="application/pdf", use_container_width=True)

def dashboard_page():
    st.title("Yönetim Paneli")
    data = db.get_all_stats()
    if data:
        df = pd.DataFrame(data, columns=['Teşhis', 'Durum'])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam", len(df))
        c2.metric("COVID", len(df[df['Teşhis']=="COVID"]))
        c3.metric("Normal", len(df[df['Teşhis']=="Normal"]))
        c4.metric("Onaylı", len(df[df['Durum']!='Bekliyor']))
        st.bar_chart(df['Teşhis'].value_counts())

def records_page():
    st.title("Hasta Arşivi")
    recs = db.get_records_by_doctor(st.session_state['username'])
    if recs: st.dataframe(pd.DataFrame(recs, columns=['ID','Dr','Hasta','Protokol','Teşhis','Skor','Tarih','Not','Durum'])[['Hasta','Teşhis','Skor','Tarih']], use_container_width=True)

def profile_page():
    st.title("Profil"); u = st.session_state['username']; d = db.get_user_profile(u)
    c1, c2 = st.columns([1,2])
    with c1: st.image(Image.open(io.BytesIO(d[3])) if d[3] else "https://via.placeholder.com/150", width=150)
    with c2: 
        if st.button("Çıkış Yap"): st.session_state['logged_in'] = False; st.rerun()

# --- MAIN ---
def main_app():
    prof = db.get_user_profile(st.session_state['username'])
    with st.sidebar:
        if prof and prof[3]: st.image(Image.open(io.BytesIO(prof[3])), width=100)
        st.markdown(f"### Dr. {st.session_state['username']}")
        if st.button("ANALİZ"): st.session_state['page'] = "Analiz"; st.rerun()
        if st.button("DASHBOARD"): st.session_state['page'] = "Dashboard"; st.rerun()
        if st.button("ARŞİV"): st.session_state['page'] = "Kayitlar"; st.rerun()
        if st.button("PROFİL"): st.session_state['page'] = "Profil"; st.rerun()
        if st.button("ÇIKIŞ"): st.session_state['logged_in'] = False; st.rerun()

    if st.session_state['page'] == "Dashboard": dashboard_page()
    elif st.session_state['page'] == "Analiz": analysis_page()
    elif st.session_state['page'] == "Kayitlar": records_page()
    elif st.session_state['page'] == "Profil": profile_page()

if st.session_state['logged_in']: main_app()
else: login_page()