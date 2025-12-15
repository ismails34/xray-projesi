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
    initial_sidebar_state="collapsed" # Girişte sidebar kapalı olsun
)

# --- 2. CSS TASARIMI (KREM & LATTE - GELECEKÇİ DOKUNUŞ) ---
st.markdown("""
<style>
    /* GENEL ARKAPLAN */
    .stApp {
        background-color: #FDFBF7 !important;
    }
    
    /* GİRİŞ KARTI (LOGIN CARD) */
    .auth-card {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(93, 64, 55, 0.08); /* Yumuşak kahve gölge */
        border: 1px solid #F0E6D2;
        text-align: center;
        margin-top: 50px;
    }
    
    /* BAŞLIK VE METİNLER */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 32px;
        color: #5D4037;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }
    
    .sub-label {
        color: #8D6E63;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 30px;
        display: block;
    }
    
    /* INPUT ALANLARI - HİZALI VE TEMİZ */
    .stTextInput input {
        background-color: #FAF9F6 !important;
        border: 1px solid #E0D6C8 !important;
        border-radius: 8px !important;
        color: #5D4037 !important;
        height: 48px !important;
        padding-left: 15px !important;
    }
    .stTextInput input:focus {
        border-color: #D4A373 !important;
        box-shadow: 0 0 0 2px rgba(212, 163, 115, 0.2) !important;
    }
    
    /* BUTONLAR */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background-color: #D4A373 !important;
        color: white !important;
        font-weight: bold;
        height: 48px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #BC8A5F !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(188, 138, 95, 0.3);
    }
    
    /* LINK GÖRÜNÜMLÜ BUTON (Tertiary) */
    button[kind="secondary"] {
        background-color: transparent !important;
        color: #8D6E63 !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 14px !important;
        margin-top: 10px;
    }
    button[kind="secondary"]:hover {
        color: #5D4037 !important;
        text-decoration: underline;
        background-color: transparent !important;
        transform: none !important;
    }
    
    /* input label gizleme (temiz görünüm için) */
    .stTextInput label {
        color: #8D6E63;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'auth_mode' not in st.session_state: st.session_state['auth_mode'] = 'login'
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'page' not in st.session_state: st.session_state['page'] = "Analiz"

# Veritabanı Başlatma
db.create_tables()
if not db.check_user_exists("admin"): db.add_user("admin", "12345")

# --- MODEL VE YARDIMCI FONKSİYONLAR (AYNI KALDI) ---
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

# --- 4. YENİ GİRİŞ SAYFASI (Future & Minimal) ---
def login_page():
    # Sayfayı ortalamak için boş kolonlar
    c_left, c_center, c_right = st.columns([1, 1.2, 1])
    
    with c_center:
        # KART TASARIMI BAŞLANGICI
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        
        # Başlık ve Alt Etiket
        st.markdown('<div class="main-title">GELECEĞE DÖNÜK</div>', unsafe_allow_html=True)
        
        # MODA GÖRE İÇERİK (Login vs Register)
        if st.session_state['auth_mode'] == 'login':
            st.markdown('<span class="sub-label">GİRİŞ PORTALI</span>', unsafe_allow_html=True)
            
            # Login Formu
            u = st.text_input("Kullanıcı Adı", placeholder="Kullanıcı Adı")
            p = st.text_input("Şifre", type="password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("GİRİŞ YAP"):
                if db.login_user(u, p):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = u
                    st.session_state['page'] = "Analiz"
                    st.rerun()
                else:
                    st.error("Kullanıcı adı veya şifre hatalı.")
            
            # Geçiş Linki
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Hesabın yok mu? Kayıt Ol", type="secondary"):
                st.session_state['auth_mode'] = 'register'
                st.rerun()
                
        else:
            # Kayıt Formu
            st.markdown('<span class="sub-label">YENİ HESAP OLUŞTUR</span>', unsafe_allow_html=True)
            
            # Kayıt Alanları
            col_name1, col_name2 = st.columns(2)
            with col_name1: name = st.text_input("Ad", placeholder="Ad")
            with col_name2: surname = st.text_input("Soyad", placeholder="Soyad")
            
            email = st.text_input("E-posta Adresi", placeholder="ornek@mail.com")
            new_u = st.text_input("Kullanıcı Adı Belirle", placeholder="Kullanıcı Adı")
            
            col_pass1, col_pass2 = st.columns(2)
            with col_pass1: np1 = st.text_input("Şifre", type="password", placeholder="••••••••")
            with col_pass2: np2 = st.text_input("Şifre Tekrar", type="password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("KAYDI TAMAMLA"):
                if np1 == np2 and new_u and email:
                    if not db.check_user_exists(new_u):
                        # Kullanıcıyı DB'ye ekle
                        db.add_user(new_u, np1)
                        # Profil bilgilerini güncelle (İsim bilgisini kaydediyoruz)
                        full_name = f"{name} {surname}"
                        db.update_user_profile(new_u, full_name, "Yeni Üye", f"İletişim: {email}", None)
                        
                        st.success("Hesap oluşturuldu! Giriş yapabilirsiniz.")
                        # Otomatik logine geç
                        st.session_state['auth_mode'] = 'login'
                        # Biraz bekleyip sayfayı yenile ki kullanıcı mesajı görsün
                        import time
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Bu kullanıcı adı zaten alınmış.")
                else:
                    st.warning("Lütfen tüm alanları doldurun ve şifrelerin eşleştiğinden emin olun.")
            
            # Geri Dönüş Linki
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Zaten hesabın var mı? Giriş Yap", type="secondary"):
                st.session_state['auth_mode'] = 'login'
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True) # Kart Bitişi

# --- 5. İÇERİK SAYFALARI (AYNI) ---
def render_sidebar():
    with st.sidebar:
        prof = db.get_user_profile(st.session_state['username'])
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if prof and prof[3]: st.image(Image.open(io.BytesIO(prof[3])), width=120)
        else: st.markdown("<div style='background-color:#E0D6C8;width:80px;height:80px;border-radius:50%;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:30px;color:#5D4037;'>👨‍⚕️</div>", unsafe_allow_html=True)
        
        doc_name = prof[0] if prof and prof[0] else st.session_state['username'].capitalize()
        doc_title = prof[1] if prof and prof[1] else "Radyoloji Uzmanı"
        st.markdown(f"<h3 style='margin-bottom:0px;color:#5D4037;'>Dr. {doc_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#8D6E63;font-size:14px;margin-top:-5px;'>{doc_title}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Analiz & Rapor", use_container_width=True): st.session_state['page'] = "Analiz"; st.rerun()
        if st.button("Yönetim Paneli", use_container_width=True): st.session_state['page'] = "Dashboard"; st.rerun()
        if st.button("Hasta Arşivi", use_container_width=True): st.session_state['page'] = "Kayitlar"; st.rerun()
        if st.button("Profil Ayarları", use_container_width=True): st.session_state['page'] = "Profil"; st.rerun()
        st.markdown("<div style='margin-top:50px;'></div>", unsafe_allow_html=True)
        if st.button("Çıkış Yap", type="secondary", use_container_width=True): 
            st.session_state['logged_in'] = False; st.rerun()

def analysis_page():
    st.markdown("## X-Ray Analiz İstasyonu")
    st.markdown("<p style='color:#8D6E63;'>Yapay zeka destekli görüntü işleme ve tanı asistanı</p>", unsafe_allow_html=True)
    col_control, col_view = st.columns([1, 2.5], gap="large")
    with col_control:
        st.markdown('<div class="medical-card"><h4>Hasta Kaydı</h4>', unsafe_allow_html=True)
        h_ad = st.text_input("Hasta Adı Soyadı")
        h_id = st.text_input("Protokol No")
        st.markdown('</div><div class="medical-card"><h4>Görüntü Filtreleri</h4>', unsafe_allow_html=True)
        con = st.slider("Kontrast", 0.5, 3.0, 1.0); br = st.slider("Parlaklık", -100, 100, 0)
        c1, c2 = st.columns(2)
        with c1: clahe = st.checkbox("CLAHE")
        with c2: inv = st.checkbox("Negatif")
        st.markdown('</div><div class="medical-card" style="text-align:center;"><h4>Görüntü Yükle</h4>', unsafe_allow_html=True)
        up = st.file_uploader("", type=['jpg','png','dcm'], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_view:
        if up:
            orig = load_image_universal(up)
            if orig:
                filt = Image.fromarray(apply_filters(orig, con, br, clahe, inv))
                c1, c2 = st.columns(2); c1.image(orig, caption="Orijinal"); c2.image(filt, caption="İşlenmiş")
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
                with col_btn2: analyze = st.button("ANALİZİ BAŞLAT", use_container_width=True)
                if analyze and h_ad:
                    with st.spinner("Analiz yapılıyor..."):
                        model = model_yukle()
                        if model:
                            img = np.expand_dims(cv2.resize(np.array(orig),(224,224))/255.0, axis=0)
                            preds = model.predict(img)[0]
                            classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                            idx = np.argmax(preds); res = classes[idx]; conf = preds[idx]*100
                            st.markdown("---"); c_r1, c_r2 = st.columns(2)
                            with c_r1:
                                st.success(f"✅ {res} (%{conf:.2f})") if res=="Normal" else st.error(f"⚠️ {res} (%{conf:.2f})")
                                db.add_record(st.session_state['username'], h_ad, h_id, res, float(conf), datetime.datetime.now().strftime("%Y-%m-%d"), "AI", "Onay")
                                st.download_button("RAPOR İNDİR", data=create_pdf(st.session_state['username'], h_ad, h_id, res, conf, "AI", datetime.datetime.now().strftime("%Y-%m-%d")), file_name="rapor.pdf", mime="application/pdf")
                            with c_r2: st.bar_chart(pd.DataFrame({"D":classes,"P":preds}).set_index("D"), color="#D4A373")
                            if res!="Normal": st.image(np.clip(cv2.resize(cv2.cvtColor(cv2.applyColorMap(np.uint8(255*make_gradcam_heatmap(img, model)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB),(224,224))*0.4+cv2.resize(np.array(orig),(224,224)),0,255).astype('uint8'), caption="Odak", width=300)

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

# --- 6. ANA AKIŞ ---
if st.session_state['logged_in']:
    render_sidebar()
    if st.session_state['page'] == "Dashboard": dashboard_page()
    elif st.session_state['page'] == "Analiz": analysis_page()
    elif st.session_state['page'] == "Kayitlar": records_page()
    elif st.session_state['page'] == "Profil": profile_page()
else:
    login_page()