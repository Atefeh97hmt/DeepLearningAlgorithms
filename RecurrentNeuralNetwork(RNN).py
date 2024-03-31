import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Create a simple RNN model
def create_rnn_model(input_shape, units):
    model = Sequential([
        SimpleRNN(units=units, input_shape=input_shape, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage
X_train = np.random.rand(100, 10, 1)  # Example training data (100 samples of sequences with 10 timesteps and 1 feature)
y_train = np.random.randint(0, 2, size=(100,))  # Example training labels (binary classification)

rnn_model = create_rnn_model(input_shape=(10, 1), units=32)
rnn_model.fit(X_train, y_train, epochs=5, batch_size=32)

# Example prediction
# X_test = np.random.rand(10, 10, 1)  # Example test data
# predictions = rnn_model.predict(X_test)
# print(predictions)
