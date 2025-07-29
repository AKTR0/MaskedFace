import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 128
CLASSES = ["correct_mask", "incorrect_mask"]
MODEL_FILE = "models/HOGSVC1/hog_svm_model.pkl"
PCA_FILE = "pca_model.pkl"
DEFAULT_TEST_PATH = "test_images"

HOG_PARAMS = {
    'orientations': 12,
    'pixels_per_cell': (6, 6),
    'cells_per_block': (3, 3),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    'feature_vector': True
}

def extract_hog_features(image, hog_params=HOG_PARAMS):
    features = hog(image, **hog_params)
    return features

def preprocess_image(image, img_size=IMG_SIZE):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    image = cv2.bilateralFilter(image, 5, 50, 50)
    
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    image = image.astype(np.float32) / 255.0
    
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    return image

def load_model(model_path, pca_path=None):
    try:
        print(f"Model yükleniyor: {model_path}")
        model = joblib.load(model_path)
        
        pca = None
        if pca_path and os.path.exists(pca_path):
            print(f"PCA yükleniyor: {pca_path}")
            pca = joblib.load(pca_path)
        
        return model, pca
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None, None

def predict_single_image(model, pca, image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, f"Görüntü yüklenemedi: {image_path}"
        
        img_processed = preprocess_image(img)
        hog_feat = extract_hog_features(img_processed)
        features = hog_feat.reshape(1, -1)
        
        if pca:
            features = pca.transform(features)
        
        prediction = model.predict(features)[0]
        confidence = model.decision_function(features)[0]
        predicted_class = CLASSES[prediction]
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, f"Tahmin hatası: {e}"

def get_image_files(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append({
                    'path': os.path.join(root, file),
                    'name': file,
                    'folder': os.path.basename(root)
                })
    
    return image_files

def test_model_on_folder(model_path, test_folder, pca_path=None, save_results=True, verbose=True):
    model, pca = load_model(model_path, pca_path)
    if model is None:
        return None
    
    if not os.path.exists(test_folder):
        print(f"Test klasörü bulunamadı: {test_folder}")
        return None
    
    image_files = get_image_files(test_folder)
    if not image_files:
        print(f"Test klasöründe görüntü dosyası bulunamadı: {test_folder}")
        return None
    
    print(f"\nTest klasörü: {test_folder}")
    print(f"Toplam görüntü sayısı: {len(image_files)}")
    print("=" * 60)
    
    results = []
    successful_predictions = 0
    
    for img_info in tqdm(image_files, desc="Test ediliyor"):
        predicted_class, confidence, error = predict_single_image(model, pca, img_info['path'])
        
        if error:
            if verbose:
                print(f"{img_info['name']}: {error}")
            continue
        
        true_class = img_info['folder'] if img_info['folder'] in CLASSES else "unknown"
        is_correct = true_class == predicted_class if true_class != "unknown" else None
        
        result = {
            'image_name': img_info['name'],
            'image_path': img_info['path'],
            'folder': img_info['folder'],
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct': is_correct
        }
        
        results.append(result)
        successful_predictions += 1
        
        if verbose:
            status = ""
            if is_correct is not None:
                status = "DOĞRU" if is_correct else "YANLIŞ"
            
            print(f"{status} {img_info['name']:<30} | Tahmin: {predicted_class:<15} | Güven: {confidence:6.3f}")
    
    print("\n" + "=" * 60)
    print(f"Başarılı tahmin: {successful_predictions}/{len(image_files)}")
    
    analyze_performance(results)
    
    if save_results:
        save_test_results(results, model_path, test_folder)
    
    return results

def analyze_performance(results):
    labeled_results = [r for r in results if r['correct'] is not None]
    
    if not labeled_results:
        print("Performans analizi yapılamadı: Gerçek etiketler bulunamadı")
        print("(Görüntüler sınıf klasörlerinde organize edilmeli)")
        return
    
    correct_predictions = [r for r in labeled_results if r['correct']]
    accuracy = len(correct_predictions) / len(labeled_results)
    
    print(f"\nPERFORMANS ANALİZİ")
    print("-" * 30)
    print(f"Genel Doğruluk: {accuracy:.4f} ({len(correct_predictions)}/{len(labeled_results)})")
    
    print(f"\nSınıf Bazlı Performans:")
    print("-" * 30)
    
    class_stats = {}
    for class_name in CLASSES:
        class_results = [r for r in labeled_results if r['true_class'] == class_name]
        if class_results:
            correct_class = [r for r in class_results if r['correct']]
            class_accuracy = len(correct_class) / len(class_results)
            class_stats[class_name] = {
                'total': len(class_results),
                'correct': len(correct_class),
                'accuracy': class_accuracy
            }
            print(f"{class_name:<20}: {class_accuracy:.4f} ({len(correct_class)}/{len(class_results)})")
        else:
            print(f"{class_name:<20}: Veri yok")
    
    if len(class_stats) == 2:
        true_labels = [r['true_class'] for r in labeled_results]
        pred_labels = [r['predicted_class'] for r in labeled_results]
        
        print(f"\nConfusion Matrix:")
        print("-" * 30)
        cm = confusion_matrix(true_labels, pred_labels, labels=CLASSES)
        
        for i, true_class in enumerate(CLASSES):
            row = " | ".join([f"{cm[i][j]:4d}" for j in range(len(CLASSES))])
            print(f"{true_class:<15} | {row}")
        
        print(f"\nGüven Skoru Analizi:")
        print("-" * 30)
        confidences = [r['confidence'] for r in labeled_results]
        correct_confidences = [r['confidence'] for r in labeled_results if r['correct']]
        wrong_confidences = [r['confidence'] for r in labeled_results if not r['correct']]
        
        print(f"Ortalama güven skoru: {np.mean(confidences):.4f}")
        if correct_confidences:
            print(f"Doğru tahminler: {np.mean(correct_confidences):.4f} (±{np.std(correct_confidences):.4f})")
        if wrong_confidences:
            print(f"Yanlış tahminler: {np.mean(wrong_confidences):.4f} (±{np.std(wrong_confidences):.4f})")

def save_test_results(results, model_path, test_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.txt"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("HOG+SVM MASK CLASSIFICATION - TEST SONUÇLARI\n")
            f.write("=" * 60 + "\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test Klasörü: {test_folder}\n")
            f.write(f"Toplam Görüntü: {len(results)}\n")
            
            labeled_results = [r for r in results if r['correct'] is not None]
            if labeled_results:
                correct_predictions = [r for r in labeled_results if r['correct']]
                accuracy = len(correct_predictions) / len(labeled_results)
                f.write(f"Doğruluk Oranı: {accuracy:.4f} ({len(correct_predictions)}/{len(labeled_results)})\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("DETAYLI SONUÇLAR\n")
            f.write("=" * 60 + "\n")
            
            for result in results:
                f.write(f"\nGörüntü: {result['image_name']}\n")
                f.write(f"Klasör: {result['folder']}\n")
                f.write(f"Gerçek Sınıf: {result['true_class']}\n")
                f.write(f"Tahmin: {result['predicted_class']}\n")
                f.write(f"Güven Skoru: {result['confidence']:.4f}\n")
                if result['correct'] is not None:
                    f.write(f"Sonuç: {'DOĞRU' if result['correct'] else 'YANLIŞ'}\n")
                f.write("-" * 40 + "\n")
        
        print(f"\nTest sonuçları kaydedildi: {results_file}")
        
    except Exception as e:
        print(f"Sonuçlar kaydedilemedi: {e}")

def main():
    parser = argparse.ArgumentParser(description='HOG+SVM Model Test')
    parser.add_argument('--test_folder', '-t', 
                       default=DEFAULT_TEST_PATH,
                       help=f'Test klasörü yolu (varsayılan: {DEFAULT_TEST_PATH})')
    parser.add_argument('--model', '-m', 
                       default=MODEL_FILE,
                       help=f'Model dosyası yolu (varsayılan: {MODEL_FILE})')
    parser.add_argument('--pca', '-p', 
                       default=PCA_FILE,
                       help=f'PCA dosyası yolu (varsayılan: {PCA_FILE})')
    parser.add_argument('--no_save', action='store_true',
                       help='Sonuçları dosyaya kaydetme')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Sessiz mod (az çıktı)')
    
    args = parser.parse_args()
    
    print("HOG+SVM MODEL TEST")
    print("=" * 50)
    
    pca_path = args.pca if os.path.exists(args.pca) else None
    if pca_path is None and args.pca == PCA_FILE:
        print("PCA dosyası bulunamadı, PCA olmadan devam ediliyor")
    
    results = test_model_on_folder(
        model_path=args.model,
        test_folder=args.test_folder,
        pca_path=pca_path,
        save_results=not args.no_save,
        verbose=not args.quiet
    )
    
    if results:
        print(f"\nTest tamamlandı! Toplam {len(results)} görüntü işlendi.")
    else:
        print("Test başarısız!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("HOG+SVM MODEL TEST")
        print("=" * 50)
        
        if not os.path.exists(MODEL_FILE):
            print(f"Model dosyası bulunamadı: {MODEL_FILE}")
            exit(1)
        
        test_folder = input(f"Test klasörü yolu (varsayılan: {DEFAULT_TEST_PATH}): ").strip()
        if not test_folder:
            test_folder = DEFAULT_TEST_PATH
        
        pca_path = PCA_FILE if os.path.exists(PCA_FILE) else None
        
        results = test_model_on_folder(
            model_path=MODEL_FILE,
            test_folder=test_folder,
            pca_path=pca_path,
            save_results=True,
            verbose=True
        )
        
        if results:
            print(f"\nTest tamamlandı! Toplam {len(results)} görüntü işlendi.")
        else:
            print("Test başarısız!")