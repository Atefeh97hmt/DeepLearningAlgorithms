import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Create an Autoencoder model
def create_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder, encoder

# Example usage
X_train = np.random.rand(100, 10)  # Example training data (100 samples with 10 features)

input_dim = X_train.shape[1]
encoding_dim = 5  # Dimension of the encoded representation

autoencoder, encoder = create_autoencoder(input_dim, encoding_dim)
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)

# Example encoding and decoding
encoded_data = encoder.predict(X_train)
decoded_data = autoencoder.predict(X_train)

# Print encoded and decoded data
print("Encoded Data:", encoded_data)
print("Decoded Data:", decoded_data)
