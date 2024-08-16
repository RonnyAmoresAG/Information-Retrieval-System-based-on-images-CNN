# Information-Retrieval-System-based-on-images-CNN
![image](https://github.com/user-attachments/assets/1d84f834-23d8-494d-bb49-4b79849126a2)

## Project Description
This project implements an image-based information retrieval system using a Convolutional Neural Network (CNN) and the ResNet50V2 model. The goal is to classify images from the Caltech-101 dataset and enable the search for similar images using a k-Nearest Neighbors (k-NN) model.

## Project Structure
The project is composed of the following main components:

1. **app.py**: This file implements the web application using Flask. It allows users to upload images and search for similar images in the dataset.
2. **features.py**: Contains the code to extract features from images using the ResNet50V2 model.
3. **Modelos Ipynb**: This folder contains Jupyter notebooks with the code for training and testing the CNN model and the image retrieval system based on k-NN.
   - `CNN_pretrained_model.ipynb`: Training the CNN using ResNet50V2.
   - `Information Retrieval System based on images.ipynb`: Implementation of the image retrieval system using k-NN.
4. **static**: Folder that contains the images used in the application, as well as the CSS files for the web interface.
5. **templates**: Contains the `index.html` file, which defines the structure of the application's web page.

## Installation
To run this project locally, follow these steps:

1. Clone this repository.
   ```bash
   git clone https://github.com/RonnyAmoresAG/Information-Retrieval-System-based-on-images-CNN.git
2. Navigate to the project directory.
   ```bash
   cd Information-Retrieval-System-based-on-images-CNN

3. Create and activate a virtual environment (optional but recommended).
    ```bash
   python -m venv venv

   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

5. Install the required dependencies using the requirements.txt file.
   ```bash
   pip install -r requirements.txt

5. Run the Flask application.
    ```bash
   python app.py

6. Usage
   -Access the web application in your browser.
   -Upload an image from your computer.
   -The system will display the most similar images found in the dataset.
   -Results
   -In the CNN_pretrained_model.ipynb notebook, it is demonstrated how the ResNet50V2 network can classify images with high accuracy. Additionally, in the Information Retrieval System based on ---images.ipynb notebook, it is shown how the k-NN based system effectively retrieves similar images.

7.Contributing
   Contributions are welcome. If you wish to contribute, please open a pull request.

8.License
   This project is licensed under the MIT License. See the LICENSE file for more details.

9.Contact
   Ronny Amores - ronny.amores@epn.edu.ec

