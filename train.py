import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2L
import matplotlib.pyplot as plt
import numpy as np

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 384, 384

# Batch size
BATCH_SIZE = 2

# Define learning rate
lr_rate = 1e-5

# Define number of epoch
num_epochs = 100

# Directory
save_dir = r''
train_directory = r''
validation_directory = r''

# Define transformations
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    brightness_range=[0.6,1.1],
    channel_shift_range=0.1)

# Create your datasets
train_data = datagen.flow_from_directory(
    directory=train_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True)

val_data = datagen.flow_from_directory(
    directory=validation_directory,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)

# Calculate the number of samples in each class
num_samples_per_class = [sum(train_data.labels == i) for i in range(len(train_data.class_indices))]

# Calculate the total number of samples
total_samples = sum(num_samples_per_class)

# Calculate the class weights
class_weights = [total_samples / (len(train_data.class_indices) * num_samples) for num_samples in num_samples_per_class]

# Print class weights
print("Class Weights:")
for class_idx, weight in enumerate(class_weights):
    class_name = list(train_data.class_indices.keys())[class_idx]
    print(f"{class_name}: {weight:.4f}")

# Convert class weight to dictionary
class_weights = {class_idx: weight for class_idx, weight in enumerate(class_weights)}

def efficient_net_v2(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Base model
    base_model = EfficientNetV2L(include_top=False, weights=None, input_shape=input_shape)
    base_model.trainable = True
    x = base_model(inputs, training=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    for width in [1024, 512]:
        x = layers.Dense(width, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

# Define your model
model = efficient_net_v2((IMG_HEIGHT, IMG_WIDTH, 3), len(train_data.class_indices))

model.summary()

# Count the total number of layers
total_layers = len(model.layers)

# Print the depth
print("Model Depth:", total_layers)

# Define your loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr_rate), metrics=['accuracy'])

# Define the learning rate range and number of iterations for the finder
start_lr = 1e-8
end_lr = 1e-3
num_iterations = train_data.samples // BATCH_SIZE

class LearningRateFinder(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, end_lr, num_iterations):
        super(LearningRateFinder, self).__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = num_iterations
        self.learning_rates = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.best_loss = np.inf
        self.iteration = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self.iteration >= self.num_iterations:
            self.model.stop_training = True
            return

        lr = self.calculate_learning_rate()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs['loss']
        self.learning_rates.append(self.calculate_learning_rate())
        self.losses.append(loss)
        self.iteration += 1

    def calculate_learning_rate(self):
        fraction = self.iteration / self.num_iterations
        lr = self.start_lr * (self.end_lr / self.start_lr) ** fraction
        return lr

# Learning rate finder
lr_finder = LearningRateFinder(start_lr, end_lr, num_iterations)
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    epochs=1,  # Run only one epoch for the learning rate finder
    callbacks=[lr_finder],
    verbose=1
)

# Plot the learning rate
plt.figure(figsize=(6, 3))
plt.plot(lr_finder.learning_rates, lr_finder.losses)
plt.title('Learning Rate Finder')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.xscale('log')
plt.savefig(os.path.join(save_dir, "learning_rate_finder.png"))
plt.close()  # Close the plot to free up memory

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_data=val_data,
        validation_steps=val_data.samples // BATCH_SIZE,
        class_weight=class_weights,
        verbose=1)

    train_loss = history.history['loss'][0]
    val_loss = history.history['val_loss'][0]
    train_acc = history.history['accuracy'][0]
    val_acc = history.history['val_accuracy'][0]

    print(f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # Save the model after each epoch
    model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.h5")
    model.save(model_path)

    # Plot the loss
    plt.figure(figsize=(6, 3))
    plt.plot([train_loss], label='Train Loss')
    plt.plot([val_loss], label='Validation Loss')
    plt.title('Loss at Epoch ' + str(epoch + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"loss_epoch_{epoch+1}.png"))
    plt.close()  # Close the plot to free up memory

    # Plot the accuracy
    plt.figure(figsize=(6, 3))
    plt.plot([train_acc], label='Train Accuracy')
    plt.plot([val_acc], label='Validation Accuracy')
    plt.title('Accuracy at Epoch ' + str(epoch + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"accuracy_epoch_{epoch+1}.png"))
    plt.close()  # Close the plot to free up memory