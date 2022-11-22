
import numpy as np

from keras import Input
from keras import Model as KerasModel
from keras.layers import Activation, Conv1D, Conv1DTranspose, BatchNormalization, Dense, Flatten, TimeDistributed, Reshape

from keras.engine.functional import Functional as KerasFunctional
from keras import backend as KerasBackend


class CNN1DAE(object):

    def __init__(self, timesteps: int, nfeatures: int, filter_dims, latent_dim: int) -> None:

        self._timesteps = timesteps
        self._nfeatures = nfeatures

        self._dims = filter_dims
        self._latent_dim = latent_dim

        self._kernel_size = 3

        self._autoencoder = None
        self._encoder = None

        self.__build()

    def __buildSimple(self, inputs) -> (KerasFunctional, KerasFunctional):

        raise NotImplementedError

    def __buildDeep(self, inputs) -> (KerasFunctional, KerasFunctional):

        # encoder
        encoded = Conv1D(self._dims[0], self._kernel_size, strides=2, padding='same')(inputs)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        for dim in self._dims[1:]:
            encoded = Conv1D(filters=dim, kernel_size=self._kernel_size, strides=2, padding='same')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Activation('relu')(encoded)

        output_dim = KerasBackend.int_shape(encoded)

        encoded = Flatten()(encoded)
        encoded = Dense(self._latent_dim, activation='relu')(encoded)
        encoded = Reshape((self._latent_dim, 1))(encoded)

        # decoder
        x = Flatten()(encoded)
        x = Dense(np.prod(output_dim[1:]))(x)
        x = Reshape((output_dim[1], output_dim[2]))(x)

        decoded = Conv1DTranspose(filters=self._dims[-1], kernel_size=self._kernel_size, strides=2, activation='relu', padding='same')(x)
        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        for dim in self._dims[-2::-1]:
            decoded = Conv1DTranspose(filters=dim, kernel_size=self._kernel_size, strides=2, activation='relu', padding='same')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Activation('relu')(decoded)

        decoded = TimeDistributed(Dense(self._nfeatures))(decoded)

        return encoded, decoded

    def __build(self) -> None:

        inputs = Input(shape=(self._timesteps, self._nfeatures))
        if len(self._dims) == 1:
            raise NotImplementedError
        else:
            encoder, decoder = self.__buildDeep(inputs)

        self._autoencoder = KerasModel(inputs, decoder, name='CNN1DAE')
        self._encoder = KerasModel(inputs, encoder, name='cnn1dae_encoder')

    def summary(self) -> None:

        self._autoencoder.summary()

    @property
    def autoencoder(self) -> KerasModel:

        return self._autoencoder

    @property
    def encoder(self) -> KerasModel:

        return self._encoder


# test
if __name__ == '__main__':

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    dims = [16, 32]
    model = CNN1DAE(timesteps=140, nfeatures=1, filter_dims=dims, latent_dim=70)
    model.summary()

    model.encoder.summary()

