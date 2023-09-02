# tensorflow-training-script

This script details the process for training an image classifier using TensorFlow and the EfficientNetV2-Large architecture. Below is a comprehensive overview of the script's components.

## 📋 Table of Contents
- [Dependencies](#-dependencies)
- [Script Breakdown](#-script-breakdown)
- [How to Run](#-how-to-run)

## 📦 Dependencies
- TensorFlow
- Matplotlib
- NumPy

## 📜 Script Breakdown

### 1. **Importing Necessary Libraries**
   Essential packages for data manipulation, model creation, training, and visualization are incorporated.

### 2. **Hyperparameters and Directories:** 
   Initial definitions such as image dimensions, batch size, learning rate, number of epochs, and directory paths.

### 3. **Data Preprocessing:**
   Image augmentation and transformations are performed using `ImageDataGenerator`.

### 4. **Dataset Creation:**
   Training and validation datasets are set up using the `flow_from_directory` method.

### 5. **Class Weight Calculation:**
   To address class imbalances, the script calculates class weights.

### 6. **EfficientNetV2 Model Definition:**
   The EfficientNetV2-Large model is instantiated with modifications to its top layers to cater to the classification task.

### 7. **Learning Rate Finder:**
   Before the main training loop, a learning rate finder is utilized to determine the optimal learning rate range.

### 8. **Training Loop:**
   The main training loop with the process of updating weights, saving models post each epoch, and visualization of loss and accuracy metrics.

### 9. **Visualization:**
   The loss and accuracy for both training and validation datasets are plotted after every epoch, and the images are saved.

## 🚀 How to Run

1. **Set Paths:**
```
save_dir = r'path_to_save_directory'
train_directory = r'path_to_training_data'
validation_directory = r'path_to_validation_data'
```
2. **Execute Script:**
```
python train.py
```

3. **Output:**
```
Trained models will be saved in the save_dir with the format 'model_epoch_{epoch_number}.h5'. Additionally, accuracy and loss plots for each epoch are saved in the same directory.
```

## ⚠️ Note
Ensure ample storage and computational capabilities, especially if you're training on high-resolution images using a comprehensive model like EfficientNetV2-Large.