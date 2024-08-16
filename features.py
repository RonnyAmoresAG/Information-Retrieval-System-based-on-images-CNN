import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Configuración del modelo y dimensiones de las imágenes

# Definir las dimensiones de las imágenes de entrada
img_height, img_width = 224, 224

# Cargar el modelo guardado en disco
# Se utiliza el modelo preentrenado ResNet50V2 para la extracción de características de las imágenes
model = load_model('resnet50v2_caltech101.h5')
print("Modelo cargado.") 

# Función para extraer características de una imagen
def extract_features(image_path, model):
    # Cargar la imagen desde la ruta proporcionada y redimensionarla al tamaño esperado por el modelo
    img = image.load_img(image_path, target_size=(img_height, img_width))
    
    # Convertir la imagen a un array numpy
    img_array = image.img_to_array(img)
    
    # Expandir las dimensiones del array para que sea compatible con la entrada del modelo
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalizar los valores de los píxeles a un rango entre 0 y 1
    img_array /= 255.
    
    # Extraer características usando el modelo preentrenado
    features = model.predict(img_array)
    
    # Aplanar las características en un vector unidimensional
    features_flattened = features.flatten()
    
    return features_flattened 

# Definir la ruta a la carpeta principal que contiene las subcarpetas con las imágenes
main_folder = '/home/ronny/Documentos/RI/Proyecto_IIB/Proyecto_Modelo_Ronny/Data/caltech-101'

# Inicializar listas para almacenar las características extraídas y las rutas de las imágenes correspondientes
features_list = []
image_paths = []

# Recorrer todas las subcarpetas dentro de la carpeta principal y extraer características de cada imagen
for subdir in os.listdir(main_folder):
    subdir_path = os.path.join(main_folder, subdir)
    
    # Asegurarse de que es una carpeta y no un archivo
    if os.path.isdir(subdir_path):
        # Recorrer todas las imágenes dentro de la subcarpeta
        for img_name in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_name)
            
            # Extraer características de la imagen y agregarlas a la lista de características
            features = extract_features(img_path, model)
            features_list.append(features)
            
            # Guardar la ruta de la imagen correspondiente
            image_paths.append(img_path)

# Convertir la lista de características en un array numpy para un manejo más eficiente
features_array = np.vstack(features_list)

# Guardar el array de características y las rutas de las imágenes en disco para uso posterior
np.save('features_array.npy', features_array)
np.save('image_paths.npy', np.array(image_paths))
print("Extracción de características completada y guardada en disco.")