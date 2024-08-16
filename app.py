from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors
import shutil

# Configuración de la aplicación Flask

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Carpeta donde se almacenarán las imágenes cargadas por el usuario
UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extensiones de archivos permitidas para la carga
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename):
    # Verifica si el archivo tiene una extensión permitida comparando su extensión con la lista de ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ruta principal ('/') que maneja tanto GET como POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verificar si hay un archivo en el request
        if 'file' not in request.files:
            return redirect(request.url)  # Redirigir al mismo URL si no hay archivo
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)  # Redirigir si no se seleccionó un archivo
        if file and allowed_file(file.filename):
            # Asegurarse de que el nombre del archivo es seguro para guardar
            filename = secure_filename(file.filename)
            # Guardar el archivo en la carpeta designada
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar la imagen cargada y realizar la búsqueda de imágenes similares
            similar_images = process_image(filepath)
            
            # Copiar las imágenes similares encontradas a la carpeta `static/images/` para su visualización
            similar_images_filenames = []
            for img in similar_images:
                img_filename = os.path.basename(img)
                shutil.copy(img, os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
                similar_images_filenames.append(img_filename)

            # Renderizar la plantilla 'index.html' con las imágenes cargadas y similares
            return render_template('index.html', filename=filename, similar_images=similar_images_filenames)
    return render_template('index.html')  # Mostrar la página principal por defecto

# Función para procesar la imagen y buscar imágenes similares
def process_image(filepath):
    # Cargar y preprocesar la imagen cargada por el usuario
    img = image.load_img(filepath, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalizar los valores de los píxeles
    
    # Extraer características de la imagen usando el modelo preentrenado
    features = model.predict(img_array)
    distances, indices = knn.kneighbors([features.flatten()])  # Buscar imágenes similares usando k-NN
    
    # Obtener las rutas de las imágenes similares
    similar_images = [image_paths[idx] for idx in indices[0]]
    
    # Imprimir las rutas de las imágenes similares para depuración
    print("Imágenes Similares Encontradas:")
    for img in similar_images:
        print(img)
    
    return similar_images  # Retornar las rutas de las imágenes similares

# Ruta para servir las imágenes desde la carpeta de carga
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Configuración inicial al iniciar la aplicación

    # Definir las dimensiones de las imágenes de entrada
    img_height, img_width = 224, 224
    
    # Cargar el modelo preentrenado guardado en disco
    model = load_model('model/resnet50v2_caltech101.h5')
    print("Modelo cargado.")
    
    # Cargar las características y rutas de las imágenes desde archivos en disco
    features_array = np.load('features_array.npy')
    image_paths = np.load('image_paths.npy')
    print("Características y rutas de imágenes cargadas desde disco.")
    
    # Crear un índice k-NN utilizando las características cargadas
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(features_array)
    print("Indexación completada.")
    
    # Iniciar la aplicación Flask
    app.run(debug=True)