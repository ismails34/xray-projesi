import sqlite3
import hashlib

# --- GÜVENLİK FONKSİYONU ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

# --- VERİTABANI BAĞLANTISI ---
def create_connection():
    conn = sqlite3.connect('hastane_veritabani.db', check_same_thread=False)
    return conn

def create_tables():
    conn = create_connection()
    c = conn.cursor()
    # Kullanıcılar
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  full_name TEXT, 
                  specialization TEXT, 
                  bio TEXT, 
                  picture BLOB)''')
    # Kayıtlar
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  doctor_name TEXT, 
                  patient_name TEXT, 
                  patient_id TEXT, 
                  diagnosis TEXT, 
                  confidence REAL, 
                  date TEXT,
                  doctor_note TEXT,
                  validation_status TEXT)''')
    conn.commit()
    conn.close()

# --- KULLANICI İŞLEMLERİ (ŞİFRELİ) ---
def check_user_exists(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =?', (username,))
    return len(c.fetchall()) > 0

def add_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    # ŞİFREYİ KRİPTOLA!
    secure_password = make_hashes(password)
    c.execute('INSERT INTO users(username, password, full_name, specialization, bio, picture) VALUES (?,?,?,?,?,?)', 
              (username, secure_password, "", "", "", None))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =?', (username,))
    data = c.fetchall()
    conn.close()
    
    # Kriptolu şifreyi kontrol et
    if data:
        if check_hashes(password, data[0][1]):
            return data
    return False

# --- DİĞER FONKSİYONLAR (AYNI KALDI) ---
def get_user_profile(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT full_name, specialization, bio, picture FROM users WHERE username =?', (username,))
    return c.fetchone()

def update_user_profile(username, full_name, specialization, bio, picture_blob):
    conn = create_connection()
    c = conn.cursor()
    if picture_blob is None:
        c.execute('''UPDATE users SET full_name=?, specialization=?, bio=? WHERE username=?''', (full_name, specialization, bio, username))
    else:
        c.execute('''UPDATE users SET full_name=?, specialization=?, bio=?, picture=? WHERE username=?''', (full_name, specialization, bio, picture_blob, username))
    conn.commit()
    conn.close()

def add_record(doctor, patient, pid, diagnosis, conf, date, note, status="Bekliyor"):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO records(doctor_name, patient_name, patient_id, diagnosis, confidence, date, doctor_note, validation_status) VALUES (?,?,?,?,?,?,?,?)', 
              (doctor, patient, pid, diagnosis, conf, date, note, status))
    conn.commit()
    conn.close()

def get_records_by_doctor(doctor_name):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM records WHERE doctor_name =?', (doctor_name,))
    return c.fetchall()

def get_all_stats():
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT diagnosis, validation_status FROM records')
    return c.fetchall()