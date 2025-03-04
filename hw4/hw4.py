import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil, pathlib
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def graph_history(history, save_path="training_history"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{save_path}_accuracy.png")  
    # Plot Loss
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_path}_loss.png")

def fetch_data():
    # Define dataset path and expected file name
    dataset_dir = "./datasets"
    dataset_zip = os.path.join(dataset_dir, "dogs-vs-cats.zip")

    # Check if dataset already exists
    if not os.path.exists(dataset_zip) and not os.path.exists(os.path.join(dataset_dir, "train")):
        print("Dataset not found. Downloading...")
        
        # Initialize and authenticate
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        api.competition_download_files("dogs-vs-cats", path=dataset_dir)
        
        print("Download complete.")
    else:
        print("Dataset already exists. Skipping download.")
    train_file = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_file):
        with zipfile.ZipFile("./datasets/dogs-vs-cats.zip", "r") as zip_ref:
            zip_ref.extractall("./datasets/")
        with zipfile.ZipFile("./datasets/train.zip", "r") as zip_ref:
            zip_ref.extractall("./datasets/")
        with zipfile.ZipFile("./datasets/test1.zip", "r") as zip_ref:
            zip_ref.extractall("./datasets/")

def preprocess_data():
    fetch_data()
    original_dir = pathlib.Path("./datasets/train")
    new_base_dir = pathlib.Path("./datasets/cats_vs_dogs_small")

    def make_subset(subset_name, start_index, end_index):
        for category in ("cat", "dog"):
            dir = new_base_dir / subset_name / category
            if(not os.path.exists(dir)):
                os.makedirs(dir)
                fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
                for fname in fnames:
                    shutil.copyfile(src=original_dir / fname,
                                    dst=dir / fname)
    make_subset("train", start_index=0, end_index=1000)
    make_subset("validation", start_index=1000, end_index=1500)
    make_subset("test", start_index=1500, end_index=2500)
    train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
    validation_dataset = image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=(180, 180),
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        new_base_dir / "test",
        image_size=(180, 180),
        batch_size=32)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )
    return train_dataset, validation_dataset, test_dataset, data_augmentation
    
def train_model():
    train_dataset, validation_dataset, test_dataset, data_augmentation = preprocess_data()
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss")
    ]
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=validation_dataset,
        callbacks=callbacks)
    test_model = keras.models.load_model(
        "convnet_from_scratch_with_augmentation.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")

def pretrained_model():
    train_dataset, validation_dataset, test_dataset, data_augmentation = preprocess_data()
    conv_base = keras.applications.convnext.ConvNeXtXLarge(
        weights="imagenet",
        include_top=False,
        input_shape=(180, 180, 3))
    conv_base.trainable = False
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.RMSprop(learning_rate=1e-6),
                metrics=["accuracy"])

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_data_augmentation.keras",
        save_best_only=True,
        monitor="val_loss")
    ]
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=validation_dataset,
        callbacks=callbacks)
    test_model = keras.models.load_model(
    "feature_extraction_with_data_augmentation.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")
    graph_history(history)

def main():
    pretrained_model()

if __name__ == "__main__":
    main()