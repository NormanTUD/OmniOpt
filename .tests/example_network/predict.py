#!/usr/bin/env python3
# This generated code is licensed under WTFPL. You can do whatever you want with it, without any restrictions.
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from termcolor import colored

def predict_single_file(file_path, model, labels):
    img = image.load_img(file_path, target_size=(40, 40))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255

    prediction = model.predict(img_array)
    max_label_idx = np.argmax(prediction)
    predicted_label = labels[max_label_idx]
    print(f"Predicted label for {file_path}: {predicted_label}")

    for i in range(0, len(prediction[0])):
        if i == max_label_idx:
            print(colored(f"{labels[i]}: {prediction[0][i]}", "green"))
        else:
            print(f"{labels[i]}: {prediction[0][i]}")

def main():
    if not os.path.exists('saved_model'):
        print("Error: 'saved_model' does not exist. Please train the model first.")
        sys.exit(1)

    labels = []
    import json

    try:
        with open('labels.json', 'r') as json_file:
            labels = json.load(json_file)
    except Exception as e:
        print("Error loading labels.json:", e)

    model = None

    try:
        model = tf.keras.models.load_model('saved_model')
    except OSError as e:
        print(colored(str(e), "red"))
        sys.exit(1)


    model.summary()

    if len(sys.argv) < 2:
        print("Usage: predict.py <file1> <file2> ...")
        sys.exit(2)

    for file_path in sys.argv[1:]:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            continue
        predict_single_file(file_path, model, labels)

if __name__ == "__main__":
    main()
