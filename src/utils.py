import numpy as np
from keras.utils import to_categorical


def normalise_img(img, percentile=99.5):

    tubulin, dapi, actin = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    def normalise_channel(channel):
        p = np.percentile(channel, percentile)
        c_min = np.min(channel)
        # clip to percentile
        channel = np.clip(channel, a_min=c_min, a_max=p)
        channel = (channel - c_min) / (p - c_min)
        return channel

    norm_channels = map(normalise_channel, [tubulin, dapi, actin])
    return np.dstack(list(norm_channels))


def balanced_generator(x_source, x_target, batch_size=128):
    while True:
        idx = np.random.randint(x_source.shape[0], size=batch_size // 2)
        x_batch_s = x_source[idx]
        idx = np.random.randint(x_target.shape[0], size=batch_size // 2)
        x_batch_t = x_target[idx]

        x_batch = np.vstack([x_batch_s, x_batch_t])
        yield x_batch, x_batch


def multi_io_generator(x_source, x_target, batch_size=128):
    while True:
        idx = np.random.randint(x_source.shape[0], size=batch_size // 2)
        x_batch_s = x_source[idx]
        idx = np.random.randint(x_target.shape[0], size=batch_size // 2)
        x_batch_t = x_target[idx]
        yield [x_batch_s, x_batch_t], [x_batch_s, x_batch_t]


def daae_generator(x_source, x_target, batch_size):
    while True:
        idx = np.random.randint(x_source.shape[0], size=batch_size // 2)
        x_batch_s = x_source[idx]

        idx = np.random.randint(x_target.shape[0], size=batch_size // 2)
        x_batch_t = x_target[idx]

        x_batch = np.vstack([x_batch_s, x_batch_t])
        d_batch = to_categorical(np.hstack([np.zeros(batch_size // 2),
                                            np.ones(batch_size // 2)]))
        yield x_batch, [x_batch, d_batch]


def one_hot_encoding(labels, nb_clusters):
    return (labels[:, None] == np.arange(nb_clusters)).astype('int8')
