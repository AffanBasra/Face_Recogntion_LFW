import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Contrastive Loss function
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Custom accuracy metric for contrastive loss
def compute_accuracy(y_true, y_pred, threshold=0.5):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < threshold, y_true.dtype)), y_true.dtype))

# Load the trained model
model_path="C:/Users/Hp/Desktop/Facial Matching/ResNetModels/20EpochsResNet/Epochs20_Resnet34_Extracted_faces_Patience5_min_val_loss_BSize16_FullyTrained_Siamese_Model"
siamese_model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss, 'compute_accuracy': compute_accuracy})
siamese_model.summary()

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float32') / 255.0
    return image

# Function to predict similarity
def predict_similarity(image_path1, image_path2):
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)
    
    # Expand dimensions to match the input shape of the model
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    
    # Predict the similarity
    prediction = siamese_model.predict([image1, image2])
    if prediction < 0:
        prediction = -prediction
    
    # Determine similarity based on a threshold
    threshold = 0.5  # You can adjust this threshold based on your validation results
    similar = prediction < threshold
    
    return similar, prediction[0][0]

class FaceMatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Match Checker")

        self.img1_path = None
        self.img2_path = None

        # Frame for images
        self.frame = tk.Frame(root)
        self.frame.pack()

        # Labels to show images
        self.label1 = tk.Label(self.frame)
        self.label1.pack(side="left")
        self.label2 = tk.Label(self.frame)
        self.label2.pack(side="right")

        # Buttons to load images
        self.btn1 = tk.Button(root, text="Load Image 1", command=self.load_image1)
        self.btn1.pack()
        self.btn2 = tk.Button(root, text="Load Image 2", command=self.load_image2)
        self.btn2.pack()

        # Button to check face match
        self.check_btn = tk.Button(root, text="Check Face Match", command=self.check_face_match)
        self.check_btn.pack()

    def load_image1(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img1_path = file_path
            img = Image.open(file_path)
            img = img.resize((150, 150), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.label1.config(image=img)
            self.label1.image = img

    def load_image2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img2_path = file_path
            img = Image.open(file_path)
            img = img.resize((150, 150), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.label2.config(image=img)
            self.label2.image = img

    def check_face_match(self):
        if self.img1_path and self.img2_path:
            similar, score = predict_similarity(self.img1_path, self.img2_path)
            result = "Yes" if similar else "No"
            messagebox.showinfo("Result", f"Are the images similar? {result}\nEmbeddings Distance: {score:.4f}")
        else:
            messagebox.showwarning("Warning", "Please load both images before checking face match.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMatchApp(root)
    root.mainloop()
