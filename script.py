import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def process_image(img_path, img_size=(28, 28)):
    # Carregar a imagem e converter para escala de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar a imagem para o tamanho especificado
    img = cv2.resize(img, img_size)
    # Normalizar os pixels para valores entre 0 e 1
    img = img / 255.0
    # Flatten a imagem para um vetor 1D
    return img.flatten()

def save_images_to_txt(class_folders, train_file, test_file, img_size=(10, 10), test_size=0.2):
    # Abrir os arquivos de saída para treino e teste
    with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
        for label, folder in enumerate(class_folders):
            # Obter todos os caminhos de imagem para a classe atual
            image_paths = [os.path.join(folder, img_name) for img_name in os.listdir(folder)]
            
            # Dividir em treino e teste
            train_paths, test_paths = train_test_split(image_paths, test_size=test_size, random_state=42)
            
            # Processar e salvar as imagens de treino
            for img_path in train_paths:
                img_vector = process_image(img_path, img_size)
                img_str = f"{label} " + ' '.join(map(str, img_vector))
                train_f.write(img_str + '\n')
            
            # Processar e salvar as imagens de teste
            for img_path in test_paths:
                img_vector = process_image(img_path, img_size)
                img_str = f"{label} " + ' '.join(map(str, img_vector))
                test_f.write(img_str + '\n')

# Caminhos para as pastas de cada classe
class_folders = [
    'PotatoPlants/Potato___Early_blight',
    'PotatoPlants/Potato___healthy',
    'PotatoPlants/Potato___Late_blight'
]

# Salve as imagens processadas em arquivos .txt
save_images_to_txt(class_folders, 'train/train_images3.txt', 'test/test_images3.txt')
