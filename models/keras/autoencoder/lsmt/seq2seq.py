
from keras import Input
from keras import Model as KerasModel
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed

from keras.engine.functional import Functional as KerasFunctional


class LSTM1DAE(object):

    def __init__(self, timesteps: int, nfeatures: int, encoder_dims) -> None:

        self._timesteps = timesteps
        self._nfeatures = nfeatures
        self._dims = encoder_dims

        self._autoencoder = None
        self._encoder = None

        self.__build()

    def __buildSimple(self, inputs) -> (KerasFunctional, KerasFunctional):

        # encoder
        encoded = LSTM(self._dims[0], return_sequences=False)(inputs)

        # decoder
        decoded = RepeatVector(self._timesteps)(encoded)
        decoded = TimeDistributed(Dense(self._nfeatures))(decoded)

        return encoded, decoded

    def __buildDeep(self, inputs) -> (KerasFunctional, KerasFunctional):

        # encoder
        encoded = LSTM(self._dims[0], return_sequences=True)(inputs)
        for dim in self._dims[1:-1]:
            encoded = LSTM(dim, return_sequences=True)(encoded)
        encoded = LSTM(self._dims[-1], return_sequences=False)(encoded)

        # decoder
        decoded = RepeatVector(self._timesteps)(encoded)
        for dim in self._dims[-2::-1]:
            decoded = LSTM(dim, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self._nfeatures))(decoded)

        return encoded, decoded

    def __build(self) -> None:

        inputs = Input(shape=(self._timesteps, self._nfeatures))
        if len(self._dims) == 1:
            encoder, decoder = self.__buildSimple(inputs)
        else:
            encoder, decoder = self.__buildDeep(inputs)

        self._autoencoder = KerasModel(inputs, decoder, name='seq2seq')
        self._encoder = KerasModel(inputs, encoder, name='seq2seq_encoder')

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

    dims = [110, 90, 70]
    model = LSTM1DAE(timesteps=140, nfeatures=1, encoder_dims=dims)
    model.summary()
