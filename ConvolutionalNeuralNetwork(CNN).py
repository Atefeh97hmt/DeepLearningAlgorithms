import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a simple CNN model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example
X_train = np.random.rand(100, 28, 28, 1)  # Example training data (100 samples of 28x28 grayscale images)
y_train = np.random.randint(0, 10, size=(100,))  # Example training labels (100 labels, 0-9)

# Reshape input data if necessary (e.g., for grayscale images)
# X_train = X_train.reshape(-1, 28, 28, 1)

cnn_model = create_cnn_model(input_shape=(28, 28, 1))
cnn_model.fit(X_train, y_train, epochs=5, batch_size=32)

# Example prediction
# X_test = np.random.rand(10, 28, 28, 1)  # Example test data
# predictions = cnn_model.predict(X_test)
# print(predictions)
