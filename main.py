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

IMG_SIZE = 128
DATASET_PATH = "dataset"
CLASSES = ["correct_mask", "incorrect_mask"]
LOG_FILE = "results_log.txt"
MODEL_FILE = "hog_svm_model_optimized.pkl"
BATCH_SIZE = 100

# İyileştirilmiş HOG parametreleri - daha zengin feature extraction
HOG_PARAMS = {
    'orientations': 12,  # 9'dan 12'ye artırıldı - daha fazla gradient yönü
    'pixels_per_cell': (6, 6),  # (8,8)'den (6,6)'ya küçültüldü - daha detaylı
    'cells_per_block': (3, 3),  # (2,2)'den (3,3)'e artırıldı - daha iyi normalizasyon
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    'feature_vector': True
}

# Satırdakiler eski parametreler
SVM_PARAMS = {
    'C': 10.0,  # 1.0 
    'max_iter': 5000,  # 2000
    'random_state': 42,
    'dual': False,
    'tol': 1e-5,  # 1e-4 
    'class_weight': 'balanced'  
}

def extract_hog_features(image, hog_params=HOG_PARAMS):
    features = hog(image, **hog_params)
    return features

def preprocess_image(image, img_size=IMG_SIZE):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) ekle
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # edge preserving
    image = cv2.bilateralFilter(image, 5, 50, 50)
    
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)  # Daha kaliteli resize
    image = image.astype(np.float32) / 255.0
    
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    return image

def load_data_efficiently(): # olmazsa çok uzun sürüyor
    all_features = []
    all_labels = []

    for label, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_PATH, class_name)
        files_list = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
        
        batch_features = []
        batch_labels = []
        
        for img_path in tqdm(files_list, desc=f"Processing {class_name}"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_processed = preprocess_image(img)
            hog_feat = extract_hog_features(img_processed)
            
            batch_features.append(hog_feat)
            batch_labels.append(label)
            
            # RAM için
            if len(batch_features) >= 500:
                all_features.extend(batch_features)
                all_labels.extend(batch_labels)
                batch_features = []
                batch_labels = []
                gc.collect()
        
        all_features.extend(batch_features)
        all_labels.extend(batch_labels)
        gc.collect()
    
    return np.array(all_features, dtype=np.float32), np.array(all_labels)  

def optimize_features(X, n_components=0.95):  # Memory için düşürüldü
    print("Memory-efficient PCA uygulanıyor...")
    
    # memory efficient
    from sklearn.decomposition import IncrementalPCA
    
    batch_size = min(1000, X.shape[0] // 10)
    
    if isinstance(n_components, float):
        sample_size = min(2000, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[sample_indices]
        
        temp_pca = PCA(random_state=42)
        temp_pca.fit(X_sample)
        
        cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= n_components) + 1
        print(f"PCA component sayısı: {n_components}")
        
        del temp_pca, X_sample
        gc.collect()
    
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        pca.partial_fit(batch)
    
    # Transform
    X_reduced = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        X_reduced.append(pca.transform(batch))
    
    X_reduced = np.vstack(X_reduced)
    
    print(f"PCA: {X.shape[1]} -> {X_reduced.shape[1]} features")
    print(f"Açıklanan varyans: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_reduced, pca

def hyperparameter_optimization(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(random_state=42, max_iter=5000))
    ])

    # Memory için daha küçük grid
    param_grid = {
        'svm__C': [1.0, 5.0, 10.0, 20.0],  # Daha az seçenek
        'svm__loss': ['hinge', 'squared_hinge'],
        'svm__dual': [False],
        'svm__tol': [1e-4, 1e-3],  # Daha az seçenek
        'svm__class_weight': ['balanced'],  # Sadece balanced
        'svm__max_iter': [5000]  # Sabit değer
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # 5
        scoring='accuracy',
        n_jobs=2,  # 2
        verbose=1,
        return_train_score=True
    )

    print("Hiperparametre optimizasyonu başlıyor...")
    grid_search.fit(X_train, y_train)
    
    print(f"En iyi parametreler: {grid_search.best_params_}")
    print(f"En iyi CV skoru: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_and_evaluate():
    print("Veri yükleniyor...")
    X, y = load_data_efficiently()
    print(f"Toplam veri: {X.shape[0]} örnek, {X.shape[1]} feature")
    
    # Memory temizliği
    gc.collect()
    
    if X.shape[1] > 2000:  # Threshold düşürüldü
        print("Memory-efficient PCA uygulanıyor...")
        X, pca = optimize_features(X, n_components=0.95)  # 0.98
        joblib.dump(pca, "pca_model.pkl")
        gc.collect()
    else:
        pca = None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Eğitim seti: {X_train.shape[0]} örnek")
    print(f"Test seti: {X_test.shape[0]} örnek")
    
    # Memory temizliği
    del X, y
    gc.collect()

    # Basit model ile karşılaştırma
    print("\nBasit model eğitiliyor...")
    simple_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(**SVM_PARAMS))
    ])
    simple_model.fit(X_train, y_train)
    simple_pred = simple_model.predict(X_test)
    simple_accuracy = accuracy_score(y_test, simple_pred)
    print(f"Basit model doğruluğu: {simple_accuracy:.4f}")

    # Memory temizliği
    del simple_model, simple_pred
    gc.collect()

    # Optimized model
    print("\nOptimize edilmiş model eğitiliyor...")
    optimized_model = hyperparameter_optimization(X_train, y_train)
    optimized_pred = optimized_model.predict(X_test)
    optimized_accuracy = accuracy_score(y_test, optimized_pred)
    
    print(f"Optimize edilmiş model doğruluğu: {optimized_accuracy:.4f}")
    print(f"İyileştirme: {optimized_accuracy - simple_accuracy:.4f}")

    # Cross-validation skorları (memory için küçültüldü)
    cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=3)
    print(f"CV skorları: {cv_scores}")
    print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\nDetaylı classification report:")
    print(classification_report(y_test, optimized_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_test, optimized_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Model kaydet
    joblib.dump(optimized_model, MODEL_FILE)
    print(f"Model kaydedildi: {MODEL_FILE}")
    
    # Log dosyasına yaz
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n{datetime.now()}\n")
        f.write(f"Basit model doğruluğu: {simple_accuracy:.4f}\n")
        f.write(f"Optimize edilmiş model doğruluğu: {optimized_accuracy:.4f}\n")
        f.write(f"İyileştirme: {optimized_accuracy - simple_accuracy:.4f}\n")
        f.write(f"CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        f.write("-" * 50 + "\n")
    
    return optimized_model

def predict_single_image(model_path, image_path, pca_path=None):
    model = joblib.load(model_path)
    pca = joblib.load(pca_path) if pca_path and os.path.exists(pca_path) else None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_processed = preprocess_image(img)
    hog_feat = extract_hog_features(img_processed)
    features = hog_feat.reshape(1, -1)
    
    if pca:
        features = pca.transform(features)

    prediction = model.predict(features)[0]
    confidence = model.decision_function(features)[0]
    result = CLASSES[prediction]
    
    print(f"Tahmin: {result}")
    print(f"Güven skoru: {confidence:.4f}")
    return result

def main():
    print("HOG+SVM Mask Classification - İyileştirilmiş Versiyon")
    print("=" * 60)
    model = train_and_evaluate()
    print("\nEğitim tamamlandı!")

if __name__ == "__main__":
    main()