#!/usr/bin/env python3

try:
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Simple neural network')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Use the trained model for prediction')
    parser.add_argument('--learning_rate', type=float, help='Learning rate as a floating point number', default=0.001)
    parser.add_argument('--epochs', type=int, help='Number of epochs as an integer', default=10)
    parser.add_argument('--validation_split', type=float, help='Validation split as a floating point number', default=0.2)
    parser.add_argument("--beta1", type=float, help="beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="beta2", default=0.999)
    parser.add_argument("--epsilon", type=float, help="epsilon", default=0.0001)
    parser.add_argument('--data', type=str, help='Data dir', default='data_tiny')
    parser.add_argument('--activation', type=str, help='Activation function (default: relu)', default='relu')
    parser.add_argument('--width', type=int, help='Width as an integer', default=40)
    parser.add_argument('--height', type=int, help='Height as an integer', default=40)
    parser.add_argument('--conv', type=int, help='Number of conv layers', default=2)
    parser.add_argument('--conv_filters', type=int, help='Number of conv filters per', default=4)
    parser.add_argument('--dense', type=int, help='Number of dense layers', default=2)
    parser.add_argument('--dense_units', type=int, help='Number of dense units per layer', default=32)
    parser.add_argument('--debug', action='store_true', help='Enables debug mode')

    args = parser.parse_args()

    from pprint import pprint

    def dier (msg):
        pprint(msg)
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"--data {args.data}: cannot be found")
        sys.exit(95)

    import resource
    import json

    import tensorflow as tf
    from keras import layers

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping
    from tensorflow.keras.optimizers import Adam

    model = tf.keras.Sequential()

    for i in range(0, args.conv):
        model.add(layers.Conv2D(
            args.conv_filters,
            (3,3),
            trainable=True,
            use_bias=True,
            activation=args.activation,
            padding="valid",
            strides=(1, 1),
            dilation_rate=(1,1),
            kernel_initializer="glorot_uniform",
            bias_initializer="variance_scaling",
            dtype="float32"
        ))

        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    for i in range(0, args.dense):
        model.add(layers.Dense(
            trainable=True,
            use_bias=True,
            units=args.dense_units,
            activation=args.activation,
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

    divide_by = 255

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
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        args.data,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    labels_array = [labels[value] for value in labels]

    try:
        with open('labels.json', 'w') as json_file:
            json.dump(labels_array, json_file)
    except Exception as e:
        print("Error writing the JSON file:", e)

    optimizer = Adam(
        beta_1=args.beta1,
        beta_2=args.beta2,
        epsilon=args.epsilon,
        learning_rate=args.learning_rate
    )

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=args.epochs, callbacks=[TerminateOnNaN(), EarlyStopping()])

    loss_obj = history.history["loss"]
    last_loss = loss_obj[len(loss_obj) - 1]

    val_accuracy = history.history["val_accuracy"][-1]

    val_loss = history.history["val_loss"][-1]

    max_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_ram_mb = max_ram_kb / 1024  # Umrechnung in MB

    print(f"RESULT: {'{:f}'.format(last_loss)}")
    print(f"LOSS: {'{:f}'.format(last_loss)}")
    print(f"VAL_ACCURACY: {val_accuracy:.4f}")
    print(f"VAL_LOSS: {val_loss:.4f}")
    print(f"RAM_USAGE: {max_ram_mb:.2f}")
except (KeyboardInterrupt):
    pass
