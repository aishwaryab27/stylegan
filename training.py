import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential

# Define parameters
rows = 28
cols = 28
channels = 1
noise_dim = 100  # Dimension of the noise vector input to the generator

# Generator function
def build_generator(noise_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(rows * cols * channels, activation='tanh'))
    model.add(Reshape((rows, cols, channels)))
    return model

# Discriminator function
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(rows, cols, channels)))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

# GAN function combining generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Set discriminator weights to non-trainable
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Function to load images from folder and preprocess
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is not None:
            img = cv2.resize(img, (rows, cols))  # Resize to generator input size
            img = img.astype('float32') / 127.5 - 1.0  # Normalize to range [-1, 1]
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
    return np.array(images)

# Function to display generated images
def display_images(generator, noise, grid_rows=4, grid_cols=4):
    n = np.random.normal(0, 1, (grid_rows * grid_cols, noise_dim))
    images = generator.predict(n)
    images = 0.5 * images + 0.5

    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(4, 4), sharey=True, sharex=True)
    count = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            axs[i, j].imshow(images[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.show()

# Training function
def train_gan(generator, discriminator, gan3, iterations, batch_size, sample_interval, image_folder):
    X_train = load_images_from_folder(image_folder)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Rescale to [-1, 1]

    half_batch = batch_size // 2

    check_points = []
    loss = []
    accuracy = []

    for iteration in range(iterations):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        true_images = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, noise_dim))
        generated_images = generator.predict(noise)

        # Labels for true and generated images
        true_labels = np.ones((half_batch, 1))
        generated_labels = np.zeros((half_batch, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(true_images, true_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, generated_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator (only GAN model)
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        misleading_targets = np.ones((batch_size, 1))

        # Train the generator
        g_loss = gan3.train_on_batch(noise, misleading_targets)

        # Print progress and save losses
        if (iteration + 1) % sample_interval == 0:
            print(f"Iteration {iteration + 1}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")
            display_images(generator, noise_dim)

            check_points.append(iteration + 1)
            loss.append(d_loss[0])
            accuracy.append(d_loss[1] * 100)

    # Plotting accuracy
    plt.figure(figsize=(15, 5))
    plt.plot(check_points, accuracy, label="Discriminator accuracy")
    plt.xticks(check_points, rotation=90)
    plt.yticks(range(0, 101, 5))
    plt.title("Discriminator Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

# Set hyperparameters
iterations = 5000
batch_size = 128
sample_interval = 1000
image_folder = 'images/'  # Replace with your image folder path

# Build and compile the models
generator = build_generator(noise_dim)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
gan3 = build_gan(generator, discriminator)
gan3.compile(loss='binary_crossentropy', optimizer=Adam())

# Train the GAN
train_gan(generator, discriminator, gan3, iterations, batch_size, sample_interval, image_folder)
