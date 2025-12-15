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
import pydicom
import os

# --- 1. SAYFA KONFİGÜRASYONU ---
st.set_page_config(
    page_title="GELECEĞE DÖNÜK - MedAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS TASARIMI (KREM & LATTE - DÜZENLİ) ---
st.markdown("""
<style>
    /* GENEL ARKAPLAN */
    .stApp {
        background-color: #FDFBF7 !important;
    }
    
    /* SOL MENÜ (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #2E2E2E !important;
    }
    
    /* KARTLAR */
    .medical-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #F0E6D2;
        margin-bottom: 20px;
    }
    
    /* GİRİŞ EKRANI KARTI */
    .auth-card {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(93, 64, 55, 0.08);
        border: 1px solid #F0E6D2;
        text-align: center;
        margin-top: 50px;
    }

    /* BAŞLIKLAR */
    h1, h2, h3, h4, h5 {
        color: #5D4037 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* INPUT ALANLARI VE ETİKETLERİ */
    .stTextInput input {
        background-color: #FAF9F6 !important;
        border: 1px solid #E0D6C8 !important;
        border-radius: 8px;
        color: #5D4037 !important;
        padding: 10px;
    }
    .stTextInput label p {
        font-size: 15px !important;
        color: #5D4037 !important;
        font-weight: 600 !important;
    }
    
    /* BUTONLAR */
    .stButton button {
        background-color: #D4A373 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        height: 45px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #BC8A5F !important;
        transform: scale(1.02);
    }

</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE & DB ---
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = 'login'
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'page' not in st.session_state: st.session_state['page'] = "Analiz"

db.create_tables()
if not db.check_user_exists("admin"): db.add_user("admin", "12345")

# --- MODEL & YARDIMCI FONKSİYONLAR ---
@st.cache_resource
def model_yukle():
    try: return tf.keras.models.load_model('yeni_coklu_model.keras')
    except: return None

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

# --- GRAD-CAM (RENKLİ HARİTA) FONKSİYONU ---
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

# --- 4. GİRİŞ SAYFASI ---
def login_page():
    c_left, c_center, c_right = st.columns([1, 1.2, 1])
    with c_center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<h1 style="color:#5D4037; font-size:36px; margin-bottom:5px;">GELECEĞE DÖNÜK</h1>', unsafe_allow_html=True)
        
        if st.session_state['auth_mode'] == 'login':
            st.markdown('<p style="color:#8D6E63; letter-spacing:2px; font-size:12px;">GİRİŞ PORTALI</p>', unsafe_allow_html=True)
            u = st.text_input("Kullanıcı Adı", placeholder="Örn: admin")
            p = st.text_input("Şifre", type="password", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("GİRİŞ YAP", use_container_width=True):
                if db.login_user(u, p):
                    st.session_state['logged_in'] = True; st.session_state['username'] = u; st.session_state['page'] = "Analiz"; st.rerun()
                else: st.error("Hatalı Kullanıcı Adı veya Şifre!")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Hesabın yok mu? Kayıt Ol", type="secondary"):
                st.session_state['auth_mode'] = 'register'; st.rerun()
        else:
            st.markdown('<p style="color:#8D6E63; letter-spacing:2px; font-size:12px;">YENİ ÜYELİK</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: name = st.text_input("Ad", placeholder="Adınız")
            with c2: surname = st.text_input("Soyad", placeholder="Soyadınız")
            new_u = st.text_input("Kullanıcı Adı Belirle", placeholder="Örn: dr_ahmet")
            p1 = st.text_input("Şifre", type="password", placeholder="Şifreniz")
            p2 = st.text_input("Şifre Tekrar", type="password", placeholder="Şifrenizi doğrulayın")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("KAYIT OL", use_container_width=True):
                if p1==p2 and new_u:
                    if not db.check_user_exists(new_u):
                        db.add_user(new_u, p1)
                        db.update_user_profile(new_u, f"{name} {surname}", "Yeni Üye", "", None)
                        st.success("Kayıt Başarılı! Giriş yapabilirsiniz."); import time; time.sleep(1.5); st.session_state['auth_mode'] = 'login'; st.rerun()
                    else: st.error("Bu kullanıcı adı zaten alınmış.")
                else: st.warning("Lütfen tüm alanları doldurun.")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Geri Dön", type="secondary"): st.session_state['auth_mode'] = 'login'; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 5. SIDEBAR ---
def render_sidebar():
    with st.sidebar:
        prof = db.get_user_profile(st.session_state['username'])
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if prof and prof[3]: st.image(Image.open(io.BytesIO(prof[3])), width=100)
        else: st.markdown("<div style='background-color:#E0D6C8;width:80px;height:80px;border-radius:50%;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:30px;color:#5D4037;'>👨‍⚕️</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#FFF !important; margin-top:10px;'>Dr. {prof[0] if prof and prof[0] else st.session_state['username']}</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Analiz & Rapor", use_container_width=True): st.session_state['page'] = "Analiz"; st.rerun()
        if st.button("Yönetim Paneli", use_container_width=True): st.session_state['page'] = "Dashboard"; st.rerun()
        if st.button("Hasta Arşivi", use_container_width=True): st.session_state['page'] = "Kayitlar"; st.rerun()
        if st.button("Profil Ayarları", use_container_width=True): st.session_state['page'] = "Profil"; st.rerun()
        st.markdown("<div style='margin-top:50px;'></div>", unsafe_allow_html=True)
        if st.button("Çıkış Yap", type="secondary", use_container_width=True): st.session_state['logged_in'] = False; st.rerun()

# --- 6. ANALİZ SAYFASI (YENİ DÜZENLİ TASARIM) ---
def analysis_page():
    st.markdown("##  Radyoloji İstasyonu")
    
    # Üst Kısım: Sol Panel (Girdiler) ve Sağ Panel (Görüntüleme Alanı)
    col_control, col_view = st.columns([1, 2.5], gap="medium") 
    
    with col_control:
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("<h5>📋 Hasta Bilgileri</h5>", unsafe_allow_html=True)
        h_ad = st.text_input("Hasta Adı Soyadı")
        h_id = st.text_input("Protokol No")
        st.markdown("---")
        st.markdown("<h5>📤 Görüntü Yükle</h5>", unsafe_allow_html=True)
        up = st.file_uploader("Röntgen Seç (DICOM/JPG/PNG)", type=['jpg','png','dcm'], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_view:
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        if up:
            orig = load_image_universal(up)
            if orig:
                # Resmi ortala
                st.columns(3)[1].image(orig, caption="Yüklenen Görüntü", use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Analiz Butonu
                if st.button("YAPAY ZEKA İLE ANALİZİ BAŞLAT ⚡", use_container_width=True):
                    if h_ad:
                        with st.spinner("AI Görüntüyü Tarıyor ve Odak Haritası Oluşturuyor..."):
                            model = model_yukle()
                            if model:
                                # 1. Tahmin Yap
                                img_arr = np.array(orig)
                                img_rez = cv2.resize(img_arr, (224,224))
                                img_fin = np.expand_dims(img_rez/255.0, axis=0)
                                preds = model.predict(img_fin)[0]
                                classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                                idx = np.argmax(preds); res = classes[idx]; conf = preds[idx]*100
                                
                                # 2. Renkli Haritayı (Grad-CAM) Oluştur
                                hm_img = np.clip(cv2.resize(cv2.cvtColor(cv2.applyColorMap(np.uint8(255*make_gradcam_heatmap(img_fin, model)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), (224,224))*0.4+img_rez,0,255).astype('uint8')

                                st.markdown("---")
                                st.markdown("### 🩺 Analiz Sonuçları")

                                # --- SONUÇLAR: GÖRSELLER YAN YANA ---
                                col_img1, col_img2 = st.columns(2)
                                with col_img1:
                                    st.image(orig, caption="Orijinal Görüntü", use_container_width=True)
                                with col_img2:
                                    st.image(hm_img, caption="Renkli AI Odak Haritası", use_container_width=True)

                                st.markdown("<br>", unsafe_allow_html=True)

                                # --- SONUÇLAR: VERİLER YAN YANA ---
                                col_res1, col_res2 = st.columns(2)
                                with col_res1:
                                    st.markdown("##### Teşhis Sonucu")
                                    if res == "Normal":
                                        st.success(f"✅ TESPİT: {res}\nGüven Skoru: %{conf:.2f}")
                                    else:
                                        st.error(f"⚠️ BULGU: {res}\nGüven Skoru: %{conf:.2f}")
                                    
                                    # Veritabanı ve PDF
                                    db.add_record(st.session_state['username'], h_ad, h_id, res, float(conf), datetime.datetime.now().strftime("%Y-%m-%d"), "AI", "Onay")
                                    pdf_data = create_pdf(st.session_state['username'], h_ad, h_id, res, conf, "AI", datetime.datetime.now().strftime("%Y-%m-%d"))
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.download_button("📄 Resmi Raporu İndir (PDF)", data=pdf_data, file_name="rapor.pdf", mime="application/pdf", use_container_width=True)

                                with col_res2:
                                    st.markdown("##### Olasılık Dağılımı")
                                    chart_data = pd.DataFrame({"Durum":classes,"Olasılık":preds})
                                    st.bar_chart(chart_data.set_index("Durum"), color="#D4A373")

                    else:
                        st.warning("Lütfen hasta adını giriniz.")
        else:
            # Boş Durum
            st.markdown("""
            <div style="text-align: center; padding: 80px; color: #8D6E63;">
                <h2>👋 Sistem Hazır</h2>
                <p>Analize başlamak için sol panelden bir röntgen görüntüsü yükleyin.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. DİĞER SAYFALAR ---
def dashboard_page():
    st.markdown("## İstatistikler"); data = db.get_all_stats()
    if data:
        df = pd.DataFrame(data, columns=['T','D']); c1,c2,c3,c4 = st.columns(4)
        c1.metric("Toplam", len(df)); c2.metric("COVID", len(df[df['T']=="COVID"])); c3.metric("Normal", len(df[df['T']=="Normal"])); c4.metric("Onaylı", len(df[df['D']!='Bekliyor']))
        st.bar_chart(df['T'].value_counts(), color="#D4A373")

def records_page():
    st.markdown("## Arşiv"); recs = db.get_records_by_doctor(st.session_state['username'])
    if recs: st.dataframe(pd.DataFrame(recs, columns=['ID','Dr','Hasta','P','Teşhis','Skor','T','N','D'])[['Hasta','Teşhis','Skor','T','D']], use_container_width=True)

def profile_page():
    st.markdown("## Profil"); u = st.session_state['username']; d = db.get_user_profile(u)
    c1, c2 = st.columns([1,2])
    with c1:
        if d and d[3]: st.image(Image.open(io.BytesIO(d[3])), width=150)
        new_pic = st.file_uploader("Fotoğraf", type=['png','jpg'])
    with c2:
        name = st.text_input("Ad Soyad", value=d[0] if d and d[0] else "")
        spec = st.text_input("Uzmanlık", value=d[1] if d and d[1] else "")
        bio = st.text_area("Bio", value=d[2] if d and d[2] else "")
        if st.button("Kaydet", type="primary"):
            blob = new_pic.getvalue() if new_pic else (d[3] if d else None)
            db.update_user_profile(u, name, spec, bio, blob)
            st.success("Kaydedildi!"); st.rerun()

# --- 8. ANA AKIŞ ---
if st.session_state['logged_in']:
    render_sidebar()
    if st.session_state['page'] == "Dashboard": dashboard_page()
    elif st.session_state['page'] == "Analiz": analysis_page()
    elif st.session_state['page'] == "Kayitlar": records_page()
    elif st.session_state['page'] == "Profil": profile_page()
else:
    login_page()