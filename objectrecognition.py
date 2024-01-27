















import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Load the trained model
model.save('cifar10_model.h5')

# Load the model for inference
model = tf.keras.models.load_model('cifar10_model.h5')

# Define class labels for CIFAR-10 dataset
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture video frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (32, 32))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Make predictions using the trained model
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions)

    # Display the predicted class on the frame
    cv2.putText(frame, class_labels[predicted_class], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
