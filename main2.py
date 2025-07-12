import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

# Ayarlar
IMG_SIZE = 128
DATASET_PATH = "dataset"
CLASSES = ["correct_mask", "incorrect_mask"]
LOG_FILE = "results_log.txt"
MODEL_FILE = "hog_svm_model.pkl"

# HOG öznitelik çıkarımı
def extract_hog_features(image):
    features, _ = hog(image,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=True)
    return features

# Veri yükleme
features = []
labels = []

print("Görüntüler yükleniyor...")
for label, class_name in enumerate(CLASSES):
    folder_path = os.path.join(DATASET_PATH, class_name)
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                hog_feat = extract_hog_features(img)
                features.append(hog_feat)
                labels.append(label)

# NumPy dizilerine çevir
X = np.array(features)
y = np.array(labels)

print(f"Toplam örnek sayısı: {len(X)}")
if len(X) == 0:
    print("Hiç görüntü yüklenemedi.")
    exit()

# Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Modeli eğit
print("Model eğitiliyor (LinearSVC)...")
model = LinearSVC()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Değerlendirme
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=CLASSES)
conf_matrix = confusion_matrix(y_test, y_pred)

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(f"Toplam örnek: {len(X)}\n")
    f.write(f"Eğitim kümesi: {X_train.shape[0]} örnek\n")
    f.write(f"Test kümesi: {X_test.shape[0]} örnek\n")
    f.write(f"Model: HOG + LinearSVC\n")
    f.write(f"Doğruluk (accuracy): {accuracy:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix) + "\n")
    f.write("=" * 60 + "\n\n")

print("Sonuçlar 'results_log.txt' dosyasına kaydedildi.")

joblib.dump(model, MODEL_FILE)
print(f"Model '{MODEL_FILE}' dosyasına kaydedildi.")
