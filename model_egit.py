import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from sklearn.utils import class_weight 

# --- ULTRA AYARLAR ---
VERI_YOLU = 'yeni_veri_seti'
IMG_BOYUT = (224, 224) 
BATCH_SIZE = 16 # Daha detaylı öğrenmesi için azalttık (Hafıza dolmasın diye)
EPOCHS = 50     # Sabırlıyız, 50 tur dönsün! (Gerekirse erken duracak)
LEARNING_RATE = 1e-5 # Çok yavaş ve hassas öğrenme hızı

print(f"🚀 ULTRA MOD: FINE-TUNING EĞİTİMİ BAŞLIYOR... ({EPOCHS} Tur)")
print("NOT: Bu işlem uzun sürecektir. Bilgisayarı kapatmayın.")

# 1. Sınıf Ağırlıkları (Veri Silmeye Gerek Yok!)
dosya_sayilari = {}
toplam_resim = 0
siniflar = sorted(os.listdir(VERI_YOLU))

print("\n📊 Veri Analizi:")
for sinif in siniflar:
    yol = os.path.join(VERI_YOLU, sinif)
    if os.path.isdir(yol):
        sayi = len(os.listdir(yol))
        dosya_sayilari[sinif] = sayi
        toplam_resim += sayi
        print(f" - {sinif}: {sayi} resim")

# Ağırlık Hesapla (Az olana çok puan)
class_weights = {}
for i, sinif in enumerate(siniflar):
    count = dosya_sayilari[sinif]
    weight = toplam_resim / (len(siniflar) * count)
    class_weights[i] = weight

print("\n⚖️  Adalet Sistemi (Class Weights) Aktif Edildi.")

# 2. Zorlu Veri Artırma (Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Daha çok döndür
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,         # Daha çok yakınlaştır (Detay görsün)
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2], # Işık değişimlerini öğrensin
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    VERI_YOLU,
    target_size=IMG_BOYUT,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    VERI_YOLU,
    target_size=IMG_BOYUT,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 3. Model Mimarisi (Fine-Tuning)
print("\n🧠 Beyin Ameliyatı Yapılıyor (Katmanlar Açılıyor)...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# KİLİT NOKTA: İlk 100 katmanı dondur, sonrasını serbest bırak (Fine-Tuning)
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) # Dengeleyici
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x) # Ezberlemeyi önle
predictions = Dense(len(siniflar), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Akıllı Takipçiler (Callbacks)
checkpoint = ModelCheckpoint('yeni_coklu_model.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, # Sadece rekor kırarsa kaydet
                             mode='max', 
                             verbose=1)

early_stop = EarlyStopping(monitor='val_loss', 
                           patience=10, # 10 tur gelişmezse dur
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=3, 
                              min_lr=1e-7, 
                              verbose=1)

# 5. Eğitimi Başlat
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("\n✅ ULTRA EĞİTİM TAMAMLANDI!")
print("En iyi model 'yeni_coklu_model.keras' olarak kaydedildi.")