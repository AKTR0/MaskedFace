import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# === 1. Parametreler ===
IMG_SIZE = 256
BATCH_SIZE = 32
MODEL_PATH = './models/CNN1/mask_classifier_model.h5'
TEST_DIR = './demodataset/10'  

# === 2. Modeli Yükle ===
model = load_model(MODEL_PATH)

# === 3. Test Verisi ===
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # 2 sınıf var
    shuffle=False
)

# === 4. Tahminler ===
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# === 5. Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# === 6. Classification Report ===
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === 7. F1-Score ===
f1 = f1_score(y_true, y_pred, average='macro')  
print(f"Macro F1 Score: {f1:.4f}")
