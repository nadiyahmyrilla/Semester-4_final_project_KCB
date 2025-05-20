# flower_app/model_knn.py
import os
import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# SETUP
dataset_path = "dataset_bunga"
resize_dim = (32, 32)
model_path = "flower_app/knn_model.pkl"
label_path = "flower_app/label_encoder.pkl"

def extract_rgb_features(image, size=resize_dim):
    image = cv2.resize(image, size)
    return image.flatten()

def train_model():
    data, labels = [], []
    imagePaths = list(paths.list_images(dataset_path))
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        if image is None:
            continue
        features = extract_rgb_features(image)
        data.append(features)
        labels.append(label)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    X_train, _, y_train, _ = train_test_split(data, labels_encoded, test_size=0.25, random_state=42)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    joblib.dump(le, label_path)

def predict_image(image):
    model = joblib.load(model_path)
    le = joblib.load(label_path)

    # Pastikan input berupa bytes dan decode dengan aman
    try:
        image_bytes = image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return "Gagal membaca gambar."

    if img_cv is None:
        return "Gambar tidak valid atau kosong."

    features = extract_rgb_features(img_cv)
    prediction = model.predict([features])[0]
    return le.inverse_transform([prediction])[0]

def test_model():
    dataset_path = "dataset_bunga"
    flower_dirs = ['coltsfoot', 'daisy', 'dandelion', 'sunflower']

    data = []
    labels = []

    for flower in flower_dirs:
        folder = os.path.join(dataset_path, flower)
        image_paths = list(paths.list_images(folder))[:10]

        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_rgb_features(image)
                data.append(features)
                labels.append(flower)

    le = LabelEncoder()
    y_true = le.fit_transform(labels)

    metrics = {}

    for k in [1, 3, 5, 7, 9]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(data, y_true)
        y_pred = model.predict(data)

        metrics[f'k_{k}'] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }

    return metrics
