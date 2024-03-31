import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Define the generator model
def build_generator(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Generate random noise for the generator
def generate_latent_points(latent_dim, n_samples):
    return np.random.normal(0, 1, (n_samples, latent_dim))

# Generate fake samples with the generator
def generate_fake_samples(generator, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    fake_samples = generator.predict(latent_points)
    return fake_samples

# Plot generated samples
def plot_generated_samples(generator, latent_dim, n_samples=100):
    fake_samples = generate_fake_samples(generator, latent_dim, n_samples)
    for i in range(n_samples):
        plt.subplot(10, 10, 1 + i)
        plt.axis('off')
        plt.imshow(fake_samples[i, :, :, 0], cmap='gray_r')
    plt.show()

# Example usage
latent_dim = 100
img_shape = (28, 28, 1)

# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, img_shape)

# Build and compile the GAN
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Train the GAN
n_epochs = 100
batch_size = 64
half_batch = batch_size // 2

# Load real samples
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_train = (X_train - 127.5) / 127.5

for epoch in range(n_epochs):
    # Train discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_samples = X_train[idx]
    labels_real = np.ones((half_batch, 1))
    d_loss_real = discriminator.train_on_batch(real_samples, labels_real)
    fake_samples = generate_fake_samples(generator, latent_dim, half_batch)
    labels_fake = np.zeros((half_batch, 1))
    d_loss_fake = discriminator.train_on_batch(fake_samples, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    latent_points = generate_latent_points(latent_dim, batch_size)
    labels_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(latent_points, labels_gan)

    # Print progress
    print(f'Epoch: {epoch + 1}, D Loss: {d_loss[0]}, Accuracy: {100 * d_loss[1]}%, G Loss: {g_loss}')

# Plot generated samples
plot_generated_samples(generator, latent_dim)
