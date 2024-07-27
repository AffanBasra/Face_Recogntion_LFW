**Siamese Neural Network for Face Recognition
Overview**
This project implements a Siamese Neural Network using ResNet-34 as the base network to perform face recognition. The model was trained on the Labeled Faces in the Wild (LFW) dataset, formatted into image pairs with labels indicating whether the pairs are of the same person or not. The preprocessing involved face extraction and noise reduction, leading to approximately 90% accuracy on validation and test sets after 20 epochs.

**Dataset**
The LFW dataset was used for training, validation, and testing. The dataset was preprocessed to:

Format images into pairs.
Assign labels (true or false) to each pair.
Extract faces from images to reduce noise.
Model Architecture
The architecture used is a Siamese Neural Network with ResNet-34 as the base network for feature extraction. The model calculates the L1 distance between the embeddings of the image pairs and uses a Dense layer with a linear activation function to predict the similarity score.

**Setup
Prerequisites**
Python 3.x
TensorFlow 2.x
NumPy
OpenCV

Installation
Clone the repository:

sh
Copy code
git clone https://github.com/AffanBasra/Face_Recogntion_LFW.git
cd siamese-face-recognition
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Dataset Preparation
Download the LFW dataset from here.
Preprocess the dataset to format it into image pairs and extract faces.
Ensure the dataset is in the following structure:
bash
Copy code
dataset/
├── train/
├── test/
└── validation/
Model Compilation and Training
Compile the Siamese ResNet-34 model:

sh
Copy code
python Siamese_resnet34_modelCompilation.py
Train the model:

sh
Copy code
python Model_training.py
The trained model will be saved as siamese_resnet34_model.h5.

Testing and Inference
Use the test script to generate embeddings and predict similarity scores for image pairs:
sh
Copy code
python Trained_ResNet_ModelTest.py --image1 path/to/image1 --image2 path/to/image2
Code Structure
Siamese_resnet34_modelCompilation.py: Script to compile the Siamese ResNet-34 model.
Model_training.py: Script to train the compiled model.
Trained_ResNet_ModelTest.py: Script to generate embeddings and predict similarity scores for image pairs using the trained model.

README.md: Project documentation.
Results
The model achieved approximately 90% accuracy on both the validation and test sets after 20 epochs of training.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
LFW Dataset
TensorFlow
