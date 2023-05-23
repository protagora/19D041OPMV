import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import *

def compact_block_diagram(model):
    fig, ax = plt.subplots(figsize=(10, 5))

    x_offset = 0
    layer_width = 50
    layer_height = 100
    layer_spacing = 20

    layer_types = {
        'Conv2D': 'Conv',
        'MaxPooling2D': 'MaxPool',
        'Flatten': 'Flatten',
        'Dense': 'Dense',
    }

    for layer in model.layers:
        layer_type = layer_types.get(layer.__class__.__name__, 'Unknown')
        rect = patches.Rectangle((x_offset, (fig.get_figheight() / 2) * fig.dpi - layer_height / 2), layer_width, layer_height, linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect)
        ax.text(x_offset + layer_width / 2, (fig.get_figheight() / 2) * fig.dpi, layer_type, fontsize=12, ha='center', va='center', rotation=0)
        x_offset += layer_width + layer_spacing

    ax.set_xlim(0, x_offset - layer_spacing)
    ax.set_ylim(0, fig.get_figheight() * fig.dpi)
    ax.axis('off')

    plt.show()

# Define the CNN architecture
def create_model():
    model = keras.Sequential([
        layers.InputLayer(input_shape=(256, 256, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3)  # output layer for yaw, pitch, and roll
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    plot_model(model, to_file='convmodel_diagram.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=96)
    return model

# Preprocess the input data
def preprocess_data(data):
    images, angles = [], []
    for image_path, angle in data:
        img = Image.open(image_path).convert('RGB').resize((256, 256))
        images.append(np.array(img))
        angles.append(angle)
    return np.array(images), np.array(angles)

# Load and preprocess the input data
data = loadAndPlotImageYpr(DATASET_PATH)

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_images, train_angles = preprocess_data(train_data)
val_images, val_angles = preprocess_data(val_data)

# Perform 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create the 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

best_val_loss = float('inf')
best_model = None

# Perform 5-fold cross-validation
for fold, (train_index, val_index) in enumerate(kfold.split(train_images)):
    print(f"Training fold {fold + 1}")
    model = create_model()

    # Example usage
    # model = create_cnn_model()
    compact_block_diagram(model)

    X_train, X_val = train_images[train_index], train_images[val_index]
    y_train, y_val = train_angles[train_index], train_angles[val_index]
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))
    
    val_loss = model.evaluate(X_val, y_val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

# Save the best model
best_model.save('models/best_model.h5')

# Evaluate the best model on the validation set
val_loss = best_model.evaluate(val_images, val_angles)
print(f"Best validation loss (MSE): {val_loss}")