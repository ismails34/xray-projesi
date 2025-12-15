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

# --- TASARIM ---
st.markdown("""
<style>
    .stApp { background-color: #FDFBF7 !important; }
    .stButton>button { background-color: #D4A373 !important; color: white !important; border-radius: 8px; }
    div[data-testid="stSidebarUserContent"] { padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- DB & GÜVENLİK ---
db.create_tables()
if not db.check_user_exists("admin"): db.add_user("admin", "12345")
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'page' not in st.session_state: st.session_state['page'] = "Analiz"

# --- HELPER FONKSİYONLAR ---
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

@st.cache_resource
def model_yukle():
    try: return tf.keras.models.load_model('yeni_coklu_model.keras')
    except: return None

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

# --- EKRANLAR ---
def login_page():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<h1 style='text-align: center; color: #5D4037;'>X-Ray Vision</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["GİRİŞ", "KAYIT"])
        with tab1:
            u = st.text_input("Kullanıcı Adı", key="l_u"); p = st.text_input("Şifre", type="password", key="l_p")
            if st.button("GİRİŞ YAP", use_container_width=True, type="primary"):
                if db.login_user(u, p):
                    st.session_state['logged_in'] = True; st.session_state['username'] = u; st.session_state['page'] = "Analiz"; st.rerun()
                else: st.error("Hatalı!")
        with tab2:
            nu = st.text_input("Yeni Kullanıcı", key="r_u"); np1 = st.text_input("Şifre", type="password", key="r_p1"); np2 = st.text_input("Tekrar", type="password", key="r_p2")
            if st.button("KAYIT OL", use_container_width=True):
                if np1==np2 and nu:
                    if not db.check_user_exists(nu): db.add_user(nu, np1); st.success("Tamam!")
                    else: st.error("Kullanılıyor.")
                else: st.error("Şifreler uyuşmuyor.")

def analysis_page():
    st.title("Radyoloji Analiz")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.info("Ayarlar"); con = st.slider("Kontrast", 0.5, 3.0, 1.0); br = st.slider("Parlaklık", -100, 100, 0)
        clahe = st.checkbox("CLAHE"); inv = st.checkbox("Negatif"); st.divider()
        h_ad = st.text_input("Hasta Adı"); up = st.file_uploader("Dosya Yükle", type=['jpg','png','dcm'])
    with c2:
        if up:
            orig = load_image_universal(up)
            if orig:
                filt = Image.fromarray(apply_filters(orig, con, br, clahe, inv))
                col1, col2 = st.columns(2); col1.image(orig, caption="Orijinal"); col2.image(filt, caption="İşlenmiş")
                if st.button("ANALİZ BAŞLAT", type="primary", use_container_width=True):
                    with st.spinner("İnceleniyor..."):
                        model = model_yukle()
                        if model:
                            img = np.expand_dims(cv2.resize(np.array(orig), (224,224))/255.0, axis=0)
                            preds = model.predict(img)[0]
                            classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                            idx = np.argmax(preds); res = classes[idx]; conf = preds[idx]*100
                            cr, cg = st.columns([1,1])
                            cr.success(f"✅ {res} (%{conf:.2f})") if res=="Normal" else cr.error(f"⚠️ {res} (%{conf:.2f})")
                            if res != "Normal": cr.image(np.clip(cv2.resize(cv2.cvtColor(cv2.applyColorMap(np.uint8(255*make_gradcam_heatmap(img, model)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), (224,224))*0.4+cv2.resize(np.array(orig), (224,224)),0,255).astype('uint8'), caption="Odak", width=250)
                            cg.bar_chart(pd.DataFrame({"S": classes, "P": preds}).set_index("S"))
                            db.add_record(st.session_state['username'], h_ad, "000", res, float(conf), datetime.datetime.now().strftime("%Y-%m-%d"), "AI", "Onay")
                            st.download_button("PDF İNDİR", data=create_pdf(st.session_state['username'], h_ad, "000", res, conf, "AI", datetime.datetime.now().strftime("%Y-%m-%d")), file_name="rapor.pdf", mime="application/pdf", use_container_width=True)

def dashboard_page():
    st.title("Panel"); data = db.get_all_stats()
    if data:
        df = pd.DataFrame(data, columns=['T','D']); c1,c2 = st.columns(2)
        c1.metric("Toplam", len(df)); c2.metric("COVID", len(df[df['T']=="COVID"]))
        st.bar_chart(df['T'].value_counts())

def records_page():
    st.title("Arşiv"); recs = db.get_records_by_doctor(st.session_state['username'])
    if recs: st.dataframe(pd.DataFrame(recs, columns=['ID','Dr','Hasta','P','Teşhis','Skor','Tarih','Not','D'])[['Hasta','Teşhis','Skor','Tarih']], use_container_width=True)

# --- BURASI DÜZELTİLEN KISIM (PROFİL) ---
def profile_page():
    st.title("Profil Düzenle")
    u = st.session_state['username']
    data = db.get_user_profile(u)
    # data[0]: Ad, data[1]: Uzmanlık, data[2]: Bio, data[3]: Resim
    
    c1, c2 = st.columns([1, 2])
    with c1:
        if data and data[3]: st.image(Image.open(io.BytesIO(data[3])), use_container_width=True)
        else: st.image("https://via.placeholder.com/150", use_container_width=True)
        new_pic = st.file_uploader("Fotoğraf Güncelle", type=['png', 'jpg'])
    
    with c2:
        name = st.text_input("Ad Soyad", value=data[0] if data and data[0] else "")
        spec = st.text_input("Uzmanlık", value=data[1] if data and data[1] else "")
        bio = st.text_area("Hakkımda", value=data[2] if data and data[2] else "")
        
        if st.button("KAYDET VE GÜNCELLE", type="primary", use_container_width=True):
            blob = new_pic.getvalue() if new_pic else (data[3] if data else None)
            db.update_user_profile(u, name, spec, bio, blob)
            st.success("Profil kaydedildi!")
            st.rerun()
            
    st.divider()
    if st.button("ÇIKIŞ YAP", type="secondary", use_container_width=True):
        st.session_state['logged_in'] = False; st.rerun()

# --- MAIN APP ---
def main_app():
    prof = db.get_user_profile(st.session_state['username'])
    with st.sidebar:
        if prof and prof[3]: st.image(Image.open(io.BytesIO(prof[3])), use_container_width=True)
        else: st.markdown("<div style='text-align:center; padding:20px; background:#eee; color:#333;'>FOTO YOK</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>Dr. {prof[0] if prof and prof[0] else st.session_state['username']}</h3>", unsafe_allow_html=True)
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