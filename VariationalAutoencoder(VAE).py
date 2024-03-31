import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Define sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Create a Variational Autoencoder model
def create_vae(input_dim, intermediate_dim, latent_dim):
    # Encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE model
    vae = Model(inputs, x_decoded_mean)

    # Compute VAE loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae

# Example usage
X_train = np.random.rand(100, 10)  # Example training data (100 samples with 10 features)

input_dim = X_train.shape[1]
intermediate_dim = 64  # Dimension of intermediate layer
latent_dim = 2  # Dimension of latent space

vae = create_vae(input_dim, intermediate_dim, latent_dim)
vae.compile(optimizer='adam')
vae.fit(X_train, epochs=10, batch_size=32)

# Example generation
# z_sample = np.random.randn(1, latent_dim)
# x_decoded = vae.decoder.predict(z_sample)
# print("Generated Data:", x_decoded)
