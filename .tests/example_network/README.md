# What is this?

This is a package there is everything to run the neural network you created in asanAI on your local hardware.

# Quickstart:

In the `data` directory, there are subdirectories, one for each category.

Put your images into these folders, and run `bash run.sh --train` to train the model. It will automatically get saved as `saved_model`, and then you can predict new images with `bash run.sh --predict filename1.jpg`.

# Files:

## run.sh

This is the run-script for the network. It installs all dependencies like TensorFlow, Keras and so on that you need to train the neural network and predict it.

## train.py

This file is for training the neural network. Run it with:

```
bash run.sh --train
```

## predict.py

This file is for predicting files with the neural network. Run it with:

```
bash run.sh --predict imagefile1.jpg imagefile2.jpg ...
```

# Problems?

Contact <norman.koch@tu-dresden.de>.
