from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.utils.safestring import mark_safe
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.stats import skew
from collections import Counter
from .model_knn import predict_image
from .model_knn import test_model

# Fungsi untuk ekstraksi fitur warna dari gambar
def extract_color_features(image_input):
    # Jika input adalah bytes, artinya ini file yang diupload oleh user
    if isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input))
    elif isinstance(image_input, str):  # path ke file
        img = Image.open(image_input)
    else:  # fallback jika file-like object (UploadedFile)
        img = Image.open(image_input)

    img = img.convert('RGB')  # Pastikan gambar dalam format RGB
    pixels = np.array(img)

    # Memisahkan channel RGB
    R = pixels[:, :, 0]
    G = pixels[:, :, 1]
    B = pixels[:, :, 2]

    # Normalisasi RGB
    R_normalized = R / 255.0
    G_normalized = G / 255.0
    B_normalized = B / 255.0

    # Menghitung rata-rata (mean), standar deviasi (std), dan skewness untuk setiap channel
    mean_R = np.mean(R_normalized)
    mean_G = np.mean(G_normalized)
    mean_B = np.mean(B_normalized)

    std_R = np.std(R_normalized)
    std_G = np.std(G_normalized)
    std_B = np.std(B_normalized)

    skew_R = skew(R_normalized.flatten())
    skew_G = skew(G_normalized.flatten())
    skew_B = skew(B_normalized.flatten())

    # Menghitung entropy untuk setiap channel
    entropy_R = entropy(R_normalized.flatten())
    entropy_G = entropy(G_normalized.flatten())
    entropy_B = entropy(B_normalized.flatten())

    return {
        'mean_R': mean_R,
        'mean_G': mean_G,
        'mean_B': mean_B,
        'std_R': std_R,
        'std_G': std_G,
        'std_B': std_B,
        'skew_R': skew_R,
        'skew_G': skew_G,
        'skew_B': skew_B,
        'entropy_R': entropy_R,
        'entropy_G': entropy_G,
        'entropy_B': entropy_B
    }

# Fungsi untuk menghitung entropy
def entropy(arr):
    counts = Counter(arr)
    total = len(arr)
    entropy_value = -sum((count/total) * np.log2(count/total) for count in counts.values())
    return entropy_value

def home(request):
    # Memuat contoh gambar dari static/images/yourimage.png
    img_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'yourimage.png')
    img_url = '/static/images/yourimage.png' if os.path.exists(img_path) else None
    return render(request, 'index.html', {
        'sample_image': img_url
    })

def dataset(request):
    base_dir = os.path.join(settings.BASE_DIR, 'dataset_bunga')
    flower_dirs = ['coltsfoot', 'daisy', 'dandelion', 'sunflower']
    
    # Data untuk setiap gambar
    flower_data = {}
    for flower in flower_dirs:
        folder = os.path.join(base_dir, flower)
        files = os.listdir(folder)[:5]
        flower_data[flower] = []

        # Mengambil gambar dan menghitung fitur RGB
        for img_file in files:
            img_path = os.path.join(folder, img_file)
            img_features = extract_color_features(img_path)

            flower_data[flower].append({
                'img': f"/dataset_bunga/{flower}/{img_file}",
                'mean_R': img_features['mean_R'],
                'mean_G': img_features['mean_G'],
                'mean_B': img_features['mean_B'],
                'std_R': img_features['std_R'],
                'std_G': img_features['std_G'],
                'std_B': img_features['std_B'],
                'skew_R': img_features['skew_R'],
                'skew_G': img_features['skew_G'],
                'skew_B': img_features['skew_B'],
                'entropy_R': img_features['entropy_R'],
                'entropy_G': img_features['entropy_G'],
                'entropy_B': img_features['entropy_B'],
                'T': 0 if flower != 'sunflower' else 1  # Contoh label, sesuaikan dengan data Anda
            })

    return render(request, 'dataset.html', {'data': flower_data})

def dataset_view(request):
    # Contoh data, sesuaikan dengan struktur data kamu
    data = {
        "coltsfoot": [...],
        "daisy": [...],
        "dandelion": [...],
        "sunflower": [...],
    }

    # Bunga pertama sebagai default aktif
    active_flower = list(data.keys())[0]  # 'coltsfoot'

    # Background berdasarkan nama bunga
    background_mapping = {
        "coltsfoot": "coltsfootBG",
        "daisy": "daisyBG",
        "dandelion": "dandelionBG",
        "sunflower": "sunflowerBG",
    }

    background = background_mapping.get(active_flower, "defaultBG")

    context = {
        "data": data,
        "background": background,
    }
    return render(request, "dataset.html", context)

# Fungsi untuk menguji dataset
def testing(request):
    try:
        # Ambil hasil evaluasi model
        metrics = test_model()
        metrics_json = {
            k: {
                'accuracy': round(v['accuracy'], 2),
                'precision': round(v['precision'], 2),
                'recall': round(v['recall'], 2)
            } for k, v in metrics.items()
        }

        # Ambil data RGB dari dataset
        base_dir = os.path.join(settings.BASE_DIR, 'dataset_bunga')
        flower_dirs = ['coltsfoot', 'daisy', 'dandelion', 'sunflower']
        flower_data = {}
        for flower in flower_dirs:
            folder = os.path.join(base_dir, flower)
            files = os.listdir(folder)[:5]
            flower_data[flower] = []

            for img_file in files:
                img_path = os.path.join(folder, img_file)
                img_features = extract_color_features(img_path)

                flower_data[flower].append({
                    'img': f"/dataset_bunga/{flower}/{img_file}",
                    'mean_R': img_features['mean_R'],
                    'mean_G': img_features['mean_G'],
                    'mean_B': img_features['mean_B'],
                    'std_R': img_features['std_R'],
                    'std_G': img_features['std_G'],
                    'std_B': img_features['std_B'],
                    'skew_R': img_features['skew_R'],
                    'skew_G': img_features['skew_G'],
                    'skew_B': img_features['skew_B'],
                    'entropy_R': img_features['entropy_R'],
                    'entropy_G': img_features['entropy_G'],
                    'entropy_B': img_features['entropy_B'],
                    'T': 0 if flower != 'sunflower' else 1
                })

        return render(request, 'testing.html', {
            'metrics': metrics_json,
            'metrics_keys': mark_safe(json.dumps(list(metrics_json.keys()))),
            'metrics_accuracy': mark_safe(json.dumps([v['accuracy'] for v in metrics_json.values()])),
            'metrics_precision': mark_safe(json.dumps([v['precision'] for v in metrics_json.values()])),
            'metrics_recall': mark_safe(json.dumps([v['recall'] for v in metrics_json.values()])),
            'data': flower_data
        })

    except Exception as e:
        return render(request, 'testing.html', {
            'error': f"Gagal memproses data testing: {e}"
        })

def klasifikasi(request):
    if request.method == 'POST' and request.FILES.get('gambar'):
        uploaded_file = request.FILES['gambar']
        try:
            # Simpan gambar ke folder media/gambar_klasifikasi/
            folder_klasifikasi = os.path.join(settings.MEDIA_ROOT, 'gambar_klasifikasi')
            os.makedirs(folder_klasifikasi, exist_ok=True)

            fs = FileSystemStorage(location=folder_klasifikasi)
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_url = fs.url(f'gambar_klasifikasi/{filename}')

            uploaded_file.seek(0)  # Reset pointer file ke awal
            hasil = predict_image(uploaded_file) # Prediksi gambar menggunakan fungsi predict_image

            return render(request, 'klasifikasi.html', {
                'hasil': hasil,
                'gambar_url': file_url
            })

        except Exception as e:
            return render(request, 'klasifikasi.html', {
                'error': f"Gagal memproses gambar: {e}"
            })

    return render(request, 'klasifikasi.html')
