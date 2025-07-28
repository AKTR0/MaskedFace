import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 128
DATASET_PATH = "dataset"
CLASSES = ["correct_mask", "incorrect_mask"]
LOG_FILE = "results_log.txt"
MODEL_FILE = "hog_svm_model_optimized.pkl"  # output ismi
PCA_FILE = "pca_model.pkl"
BATCH_SIZE = 100

HOG_PARAMS = {
    'orientations': 12,
    'pixels_per_cell': (6, 6),
    'cells_per_block': (3, 3),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    'feature_vector': True
}

SVM_PARAMS = {
    'C': 1.0,
    'max_iter': 2000,
    'random_state': 42,
    'dual': False,
    'tol': 1e-4
}

def extract_hog_features(image, hog_params=HOG_PARAMS):
    features = hog(image, **hog_params)
    return features

def preprocess_image(image, img_size=IMG_SIZE):
    """Test koduyla uyumlu ön işleme"""
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Bilateral filter - edge preserving
    image = cv2.bilateralFilter(image, 5, 50, 50)
    
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    image = image.astype(np.float32) / 255.0
    
    # Genişletme
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    return image

def load_data_efficiently():
    all_features = []
    all_labels = []

    for label, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_PATH, class_name)
        files_list = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
        for img_path in tqdm(files_list, desc=f"Processing {class_name}"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_processed = preprocess_image(img)
            hog_feat = extract_hog_features(img_processed)
            all_features.append(hog_feat)
            all_labels.append(label)
    return np.array(all_features), np.array(all_labels)

def optimize_features(X, n_components=0.95):
    print(f"PCA uygulanıyor: {X.shape} -> ", end="")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    print(f"{X_reduced.shape}")
    
    # Model Kaydet
    joblib.dump(pca, PCA_FILE)
    print(f"PCA modeli kaydedildi: {PCA_FILE}")
    
    return X_reduced, pca

def hyperparameter_optimization(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(random_state=42, max_iter=3000))
    ])

    param_grid = {
        'svm__C': [0.1, 1.0, 10.0],
        'svm__loss': ['hinge', 'squared_hinge'],
        'svm__dual': [False],
        'svm__tol': [1e-4, 1e-3]
    }

    print("Hiperparametre optimizasyonu başlıyor...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print(f"En iyi parametreler: {grid_search.best_params_}")
    print(f"En iyi skor: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate():
    print("Veri yükleniyor...")
    X, y = load_data_efficiently()
    print(f"Toplam örnek sayısı: {X.shape[0]}")
    print(f"Özellik boyutu: {X.shape[1]}")
    
    pca = None
    if X.shape[1] > 1000:
        X, pca = optimize_features(X, n_components=0.95)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Eğitim seti: {X_train.shape[0]}, Test seti: {X_test.shape[0]}")

    # Basit model
    print("\nBasit model eğitiliyor...")
    simple_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(**SVM_PARAMS))
    ])
    simple_model.fit(X_train, y_train)
    simple_pred = simple_model.predict(X_test)
    simple_accuracy = accuracy_score(y_test, simple_pred)
    print(f"Basit model doğruluğu: {simple_accuracy:.4f}")

    # Optimize model
    print("\nOptimize model eğitiliyor...")
    optimized_model = hyperparameter_optimization(X_train, y_train)
    optimized_pred = optimized_model.predict(X_test)
    optimized_accuracy = accuracy_score(y_test, optimized_pred)
    print(f"Optimize model doğruluğu: {optimized_accuracy:.4f}")

    # Cross validation
    cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=3)
    print(f"Cross validation skorları: {cv_scores}")
    print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Modeli kaydet
    joblib.dump(optimized_model, MODEL_FILE)
    print(f"Model kaydedildi: {MODEL_FILE}")
    
    # Detaylı rapor
    print("\nDetaylı sınıflandırma raporu:")
    print(classification_report(y_test, optimized_pred, target_names=CLASSES))
    
    return optimized_model, pca

def predict_single_image(model_path, image_path, pca_path=None):
    model = joblib.load(model_path)
    pca = joblib.load(pca_path) if pca_path and os.path.exists(pca_path) else None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return None
        
    img_processed = preprocess_image(img)
    hog_feat = extract_hog_features(img_processed)
    features = hog_feat.reshape(1, -1)
    
    if pca:
        features = pca.transform(features)

    prediction = model.predict(features)[0]
    confidence = model.decision_function(features)[0]
    result = CLASSES[prediction]
    print(f"Tahmin: {result} (güven: {confidence:.4f})")
    return result

def main():
    print("HOG+SVM Mask Classification - Eğitim")
    print("="*50)
    
    model, pca = train_and_evaluate()
    
    print("\nEğitim tamamlandı!")
    print(f"Model dosyası: {MODEL_FILE}")
    if pca:
        print(f"PCA dosyası: {PCA_FILE}")

if __name__ == "__main__":
    main()