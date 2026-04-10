import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import graycomatrix, graycoprops
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --------------------------
#  FEATURE EXTRACTION
# --------------------------

def extract_features(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (128, 128))
    img = cv2.equalizeHist(img)

    img_scaled = (img / 8).astype(np.uint8)  # GLCM scaling
    glcm = graycomatrix(img_scaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.extend(graycoprops(glcm, prop).flatten())

    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu)

    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Cannot read image in color: {image_path}")
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, (32, 32))
    features.extend(np.histogram(hsv[:, :, 0], bins=32, density=True)[0])

    return np.array(features)

# --------------------------
#  READ DATASET FOLDERS
# --------------------------

dataset_path = r"C:\Users\sahan\Downloads\archive (1)" # Update path

# Detect classes with at least one image
classes = []
for cls in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, cls)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if len(images) == 0:
        continue
    classes.append(cls)

print("Detected Classes:", classes)

X, y = [], []

print("Extracting features from dataset...")

for class_id, cls in enumerate(classes):
    class_path = os.path.join(dataset_path, cls)
    print(f"Processing: {cls}")

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(class_id)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

X, y = np.array(X), np.array(y)
print(f"Total Samples: {len(X)}")
print(f"Feature Dimension: {X.shape[1]}")

# --------------------------
#  TRAIN / TEST SPLIT
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
#  TRAIN SVM MODEL
# --------------------------

svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, svm_model.predict(X_train))
test_acc = accuracy_score(y_test, svm_model.predict(X_test))

print(f"\nTRAIN ACCURACY: {train_acc*100:.2f}%")
print(f"TEST ACCURACY: {test_acc*100:.2f}%\n")

# --------------------------
#  CLASSIFICATION REPORT
# --------------------------

y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes, labels=range(len(classes))))

# --------------------------
#  CONFUSION MATRIX
# --------------------------

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# --------------------------
#  SAVE MODEL
# --------------------------

os.makedirs("models", exist_ok=True)

svm_data = {
    'model': svm_model,
    'scaler': scaler,
    'classes': classes,
    'feature_dim': X.shape[1]
}

with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_data, f)

print("\n✅ MODEL SAVED: models/svm_model.pkl")
print("Classes:", classes)