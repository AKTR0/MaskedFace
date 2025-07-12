import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Ayarlar
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset"

# Eğitim ve doğrulama verilerini ayır (20% validasyon)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  
)

# CNN Modeli
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 sınıf: correct_mask, incorrect_mask
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Model eğitimi
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# Model kaydı
model.save("mask_classifier_model.h5")
print("Model başarıyla kaydedildi!")

# Validation seti üzerinde tahmin
val_gen.reset()
pred_probs = model.predict(val_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_gen.classes

# Classification report ve confusion matrix
class_labels = list(val_gen.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels)
conf_matrix = confusion_matrix(y_true, y_pred)

# Sonuçları log dosyasına yazma
log_file = "training_log.txt"

with open(log_file, "a", encoding="utf-8") as f:
    f.write("="*50 + "\n")
    f.write(f"Epoch sayısı: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Train örnek sayısı: {train_gen.samples}\n")
    f.write(f"Validation örnek sayısı: {val_gen.samples}\n\n")

    f.write("Epoch bazında eğitim doğruluğu:\n")
    for i, acc in enumerate(history.history['accuracy']):
        f.write(f"  Epoch {i+1}: {acc:.4f}\n")

    f.write("\nEpoch bazında doğrulama doğruluğu:\n")
    for i, val_acc in enumerate(history.history['val_accuracy']):
        f.write(f"  Epoch {i+1}: {val_acc:.4f}\n")

    f.write("\nEpoch bazında eğitim kaybı:\n")
    for i, loss in enumerate(history.history['loss']):
        f.write(f"  Epoch {i+1}: {loss:.4f}\n")

    f.write("\nEpoch bazında doğrulama kaybı:\n")
    for i, val_loss in enumerate(history.history['val_loss']):
        f.write(f"  Epoch {i+1}: {val_loss:.4f}\n")

    f.write("\nValidation Set - Classification Report:\n")
    f.write(report + "\n")

    f.write("\nValidation Set - Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix) + "\n")

    f.write("="*50 + "\n\n")

print(f"Eğitim sonuçları ve detaylı metrikler '{log_file}' dosyasına kaydedildi.")

# Eğitim doğruluk grafiği
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Model Accuracy")
plt.show()
