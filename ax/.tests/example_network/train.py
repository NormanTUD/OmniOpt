#!/usr/bin/env python3
# This generated code is licensed under WTFPL. You can do whatever you want with it, without any restrictions.
# python3 -m venv asanaienv
# source asanaienv/bin/activate
# pip3 install tensorflow tensorflowjs protobuf  scikit-image opencv-python 
try:
    import sys
    import os
    # python3 nn.py file_1.jpg file_2.jpg file_3.jpg
    import keras
    import tensorflow as tf
    import uuid
    import argparse

    model = tf.keras.Sequential()

    parser = argparse.ArgumentParser(description='Simple neural network')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Use the trained model for prediction')
    parser.add_argument('--learning_rate', type=float, help='Learning rate as a floating point number', default=0.001)
    parser.add_argument('--epochs', type=int, help='Number of epochs as an integer', default=10)
    parser.add_argument('--validation_split', type=float, help='Validation split as a floating point number', default=0.2)
    parser.add_argument('--data', type=str, help='Data dir', default='data')
    parser.add_argument('--width', type=int, help='Width as an integer', default=40)
    parser.add_argument('--height', type=int, help='Height as an integer', default=40)
    parser.add_argument('--debug', action='store_true', help='Enables debug mode (set -x)')

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"--data {args.data}: cannot be found")
        sys.exit(95)

    from pprint import pprint
    def dier (msg):
        pprint(msg)
        sys.exit(1)

    from keras import layers
    model.add(layers.Conv2D(
        4,
        (3,3),
        trainable=True,
        use_bias=True,
        activation="relu",
        padding="valid",
        strides=(1, 1),
        dilation_rate=(1,1),
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling",
        dtype="float32"
    ))
    model.add(layers.Conv2D(
        1,
        (3,3),
        trainable=True,
        use_bias=True,
        activation="relu",
        padding="valid",
        strides=(1, 1),
        dilation_rate=(1,1),
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling",
        dtype="float32"
    ))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        trainable=True,
        use_bias=True,
        units=32,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling",
        dtype="float32"
    ))
    model.add(layers.Dense(
        trainable=True,
        use_bias=True,
        units=len([name for name in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, name))]),
        activation="softmax",
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling",
        dtype="float32"
    ))
    model.build(input_shape=[None, args.height, args.width, 3])

    model.summary()

    from termcolor import colored

    divide_by = 255


    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Define size of images
    target_size = (args.height, args.width)

    # Create ImageDataGenerator to read images and resize them
    datagen = ImageDataGenerator(rescale=1./divide_by, # Normalize (from 0-255 to 0-1)
                                 validation_split=args.validation_split, # Split into validation and training datasets
                                 preprocessing_function=lambda x: tf.image.resize(x, target_size)) # Resize images

    # Read images and split them into training and validation dataset automatically
    train_generator = datagen.flow_from_directory(
        args.data,
        target_size=target_size,
        batch_size=10,
        class_mode='categorical',
        subset='training')

    validation_generator = datagen.flow_from_directory(
        args.data,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation')

    import json

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    labels_array = [labels[value] for value in labels]

    try:
        with open('labels.json', 'w') as json_file:
            json.dump(labels_array, json_file)
    except Exception as e:
        print("Error writing the JSON file:", e)




    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=0.0001, learning_rate=args.learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=args.epochs)

    loss_obj = history.history["loss"]
    last_loss = loss_obj[len(loss_obj) - 1]
    print(f"RESULT: {'{:f}'.format(last_loss)}")
except (KeyboardInterrupt) as e:
    pass
