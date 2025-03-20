import tensorflow as tf
from tensorflow.keras.models import Sequential  # Correct import
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Fix layer names
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r"C:\Users\Pc\OneDrive\Desktop\cat and dog\train"
val_dir = r"C:\Users\Pc\OneDrive\Desktop\cat and dog\validation"

# Image Data Generation 
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)  # Fix rescale value
val_datagen = ImageDataGenerator(rescale=1./255)

# Load images 
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),  # Fix layer name
    MaxPooling2D(2,2),  # Fix layer name
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of Model
model.summary()
