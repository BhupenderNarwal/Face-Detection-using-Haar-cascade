import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded image from the POST request
    image = request.files['image']

    # Save the uploaded image temporarily
    image_path = 'static/temp.jpg'
    image.save(image_path)

    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Generate a unique filename for the processed image
    filename = 'static/detected_faces.jpg'

    # Save the processed image
    cv2.imwrite(filename, img)

    # Delete the temporary uploaded image
    os.remove(image_path)

    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
