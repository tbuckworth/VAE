from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np

# Set the size of the latent space
latent_dim = 2

# Define the encoder network
inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)


# Define the sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Use the sampling function to get a latent space vector
z = Lambda(sampling)([z_mean, z_log_var])

# Define the decoder network
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(128, activation='relu')(decoder_inputs)
x = Dense(256, activation='relu')(x)
outputs = Dense(784, activation='sigmoid')(x)

# Define the VAE model
vae = Model(inputs, outputs)

# Define the encoder model
encoder = Model(inputs, z_mean)

# Define the generator model
decoder = Model(decoder_inputs, outputs)

# Define the VAE loss function
reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=-1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the VAE model
vae.compile(optimizer='adam')

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Train the VAE model
vae.fit(x_train,
        epochs=50,
        batch_size=128,
        validation_data=(x_test, None))

# Generate some new digits using the generator model
random_digits = np.random.normal(size=(10, latent_dim))
generated_digits = decoder.predict(random_digits)

# Display the generated digits
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original digits
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstructed digits
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoder.predict(encoder.predict(x_test[i].reshape(1, 784))).reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
