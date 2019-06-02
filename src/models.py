from keras.models import Model
from keras.layers import Layer, Input, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Conv2DTranspose, UpSampling2D, Reshape
from keras.layers import Concatenate, BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import sgd, rmsprop
from keras import backend as K


class GradientReversalLayer(Layer):

    def __init__(self, output_dim, **kwargs):

        self.output_dim = output_dim
        super(GradientReversalLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        super(GradientReversalLayer, self).build(input_shape)

    def call(self, x):

        return -x + K.stop_gradient(2 * x)

    def compute_output_shape(self, input_shape):

        return input_shape


def conv_autoencoder(input_shape=(40, 40, 3), l=1e-2):

    input_img = Input(shape=input_shape)

    x = Conv2D(16, (5, 5), padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=l2(l))(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=l2(l))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(128, kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(x)
    x = BatchNormalization()(x)
    encoding = Activation('relu')(x)

    encoder = Model(inputs=input_img, outputs=encoding)

    x = Dense(800, kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(encoding)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((10, 10, 8))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=l2(l))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x_main = Conv2D(3, (5, 5), padding='same',name='main',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(l))(x)

    model = Model(inputs=input_img, outputs=x_main)

    model.compile(optimizer='rmsprop', loss='mse')

    return model, encoder


def adversarial_conv_autoencoder(input_shape=(40, 40, 3), l=1e-2, lam=2):

    input_img = Input(shape=input_shape)

    x = Conv2D(16, (5, 5), padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=l2(l))(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=l2(l))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(128, kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(x)
    x = BatchNormalization()(x)
    encoding = Activation('relu')(x)

    encoder = Model(inputs=input_img, outputs=encoding)

    # gradient reversal layer
    x = GradientReversalLayer(64)(encoding)

    x_aux = Dense(2, activation='softmax', name='aux',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=l2(l))(x)

    x = Dense(800, kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(encoding)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((10, 10, 8))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=l2(l))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x_main = Conv2D(3, (5, 5), padding='same', name='main',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(l))(x)

    losses = {'main' : 'mse', 'aux' : 'categorical_crossentropy'}
    loss_weights = {'main' : 1., 'aux' : lam}

    model = Model(inputs=input_img, outputs=[x_main, x_aux])

    model.compile(optimizer='rmsprop', loss=losses, loss_weights=loss_weights)

    return model, encoder


def autoencoder(input_dim, dim_encoder, l=1e-3):

    x_input = Input(shape=(input_dim,))

    x = Dense(dim_encoder,
              kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    encoder = Model(inputs=x_input, outputs=x)

    x_output = Dense(input_dim,
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=l2(l))(x)

    model = Model(inputs=x_input, outputs=x_output)

    model.compile(optimizer='rmsprop', loss='mse')

    return model, encoder


def multitask_autoencoder(input_dim, nb_hidden, l=1e-3):

    # separate inputs
    x_input_s = Input(shape=(input_dim,))
    x_input_t = Input(shape=(input_dim,))
    
    # shared layer
    dense_shared = Dense(nb_hidden, activation='relu',
                         kernel_initializer='glorot_uniform',
                         kernel_regularizer=l2(l))
    encoding_s = dense_shared(x_input_s)
    encoding_t = dense_shared(x_input_t)

    # separate outputs (tasks)
    x_s = Dense(input_dim, name='source',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(l))(encoding_s)
    x_t = Dense(input_dim, name='target',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(l))(encoding_t)

    encoder = Model(inputs=x_input_s, outputs=encoding_s)
    model = Model(inputs=[x_input_s, x_input_t], outputs=[x_s, x_t])
    model.compile(loss='mse', optimizer='rmsprop')

    return model, encoder


def domain_adversarial_autoencoder(input_shape, nb_hidden, lam, l=1e-3):

    x_input = Input(shape=(input_shape,))

    x = Dense(nb_hidden,
              kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(x_input)
    x = BatchNormalization()(x)
    encoding = Activation('relu')(x)

    encoder = Model(inputs=x_input, outputs=encoding)

    x_main = Dense(input_shape, name='main',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=l2(l))(encoding)

    # gradient reversal layer
    x = GradientReversalLayer(nb_hidden)(encoding)

    x_aux = Dense(2, activation='softmax', name='aux',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=l2(l))(x)

    model = Model(inputs=[x_input], outputs=[x_main, x_aux])

    losses = {'main' : 'mse', 'aux' : 'categorical_crossentropy'}
    loss_weights = {'main' : 1., 'aux' : lam}

    opt = rmsprop(lr=1e-3, rho=0.9, epsilon=None, decay=1e-4)

    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)

    return model, encoder


def shared_layer(f_layer, x_input_s, x_input_t):
    x_s = f_layer(x_input_s)
    x_t = f_layer(x_input_t)
    return x_s, x_t


def multitask_conv_autoencoder(input_shape=(40, 40, 3), l=1e-3):

    # encoder

    x_input_s = Input(shape=input_shape)
    x_input_t = Input(shape=input_shape)

    x_s, x_t = shared_layer(Conv2D(16, (5, 5), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_regularizer=l2(l)),
                            x_input_s, x_input_t)
    x_s, x_t = shared_layer(BatchNormalization(), x_s, x_t)
    x_s, x_t = shared_layer(Activation('relu'), x_s, x_t)
    
    x_s, x_t = shared_layer(MaxPooling2D((2, 2), padding='same'), x_s, x_t)

    x_s, x_t = shared_layer(Conv2D(8, (3, 3), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_regularizer=l2(l)),
                            x_s, x_t)
    x_s, x_t = shared_layer(BatchNormalization(), x_s, x_t)
    x_s, x_t = shared_layer(Activation('relu'), x_s, x_t)

    x_s, x_t = shared_layer(MaxPooling2D((2, 2), padding='same'), x_s, x_t)

    x_s, x_t = shared_layer(Flatten(), x_s, x_t)
    x_s, x_t = shared_layer(Dense(128, kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(l)),
                                  x_s, x_t)
    x_s, x_t = shared_layer(BatchNormalization(), x_s, x_t)
    encoding_s, encoding_t = shared_layer(Activation('relu'), x_s, x_t)

    encoder = Model(inputs=x_input_s, outputs=encoding_s)
    
    # source decoder

    x_s = Dense(800, kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(l))(encoding_s)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s)

    x_s = Reshape((10, 10, 8))(x_s)
    x_s = UpSampling2D((2, 2))(x_s)
    x_s = Conv2D(16, (3, 3), padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(l))(x_s)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s)

    x_s = UpSampling2D((2, 2))(x_s)
    x_s = Conv2D(3, (5, 5), padding='same', name='main_s',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=l2(l))(x_s)

    # target decoder

    x_t = Dense(800, kernel_initializer='glorot_uniform',
              kernel_regularizer=l2(l))(encoding_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)

    x_t = Reshape((10, 10, 8))(x_t)
    x_t = UpSampling2D((2, 2))(x_t)
    x_t = Conv2D(16, (3, 3), padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=l2(l))(x_t)
    x_t = BatchNormalization()(x_t)
    x_t = Activation('relu')(x_t)

    x_t = UpSampling2D((2, 2))(x_t)
    x_t = Conv2D(3, (5, 5), padding='same', name='main_t',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=l2(l))(x_t)

    model = Model(inputs=[x_input_s, x_input_t], outputs=[x_s, x_t])

    model.compile(optimizer='rmsprop', loss='mse')

    return model, encoder
