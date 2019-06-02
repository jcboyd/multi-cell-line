from __future__ import print_function
from __future__ import division

import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.morphology import erosion, disk
from skimage.transform import rescale
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns


palette = [
    '#ff91a4', '#ffec91', '#b591ff', '#91ffb5', '#ffb591', '#cccaaa',
    '#9bd5e0', '#839485', '#5245ce', '#f82831', '#2c292d', '#e7b270',
    '#f36a46', '#f4953b', '#b3b3b3', '#977c7c', '#66c88f', '#7f8778',
    '#ce491c', '#d1854b', '#76a1db', '#033e7b', '#8c9767', '#c3a2a1',
    '#5b6c7f', '#e86666', '#303030', '#ffd5f0', '#00f9ff', '#123456']


class ImageCropper:
    """Utility class for getting groups of bounding boxes"""
    def __init__(self, well, field, ch5_folder):
        """Initialises ImageCropper

        Keyword arguments:
        well -- well of interest
        field -- field of interest
        ch5_folder -- root directory of ch5 data
        plate -- plate of interest
        """
        self.well = well
        self.field = field
        self.file = os.path.join(ch5_folder, '%s_0%s.ch5' % (well, field))

        f = h5py.File(self.file, 'r+')

        self.plate = list(f['sample/0/plate'].keys())[0]

        self.properties = f[
            'sample/0/plate/' + \
            '%s/experiment/' % self.plate + \
            '%s/position/%s/' % (self.well, self.field)]

    def get_bounding_box_list(self,
        channel='tertiary__expanded',
        category='bounding_box'):
        """Retrieves bounding box coordinates from cellh5 file"""

        bb = self.properties['feature/%s/%s' % (channel, category)]
        return bb

    def plot_image(self, ax):
        """Plots image channels together as dstack"""
        img = self.get_image()
        ax.imshow(img)
        ax.axis('off')

    def normalise_channel(self, channel):
        # calculate 99th percentile
        channel_min = np.min(channel)
        channel_p99 = np.percentile(channel, 99.9)#99.5)
        # clip extrema
        channel[channel > channel_p99] = channel_p99
        # normalise
        norm_cy5 = (channel - channel_min) / float(channel_p99 - channel_min)
        # rescale to 8-bit image
        norm_channel = np.floor(norm_cy5 * 255).astype('uint8')
        return norm_channel

    def get_contours(self, labelled_mask):
        labelled_mask[labelled_mask > 0] = 255
        eroded_mask = erosion(labelled_mask, selem=disk(3))
        contours = labelled_mask - eroded_mask
        return contours

    def get_masks(self):
        return self.properties['image/region'].value.reshape((3, 1040, 1392))

    def get_mask_crop(self, bounding_box, padding=0):
        primary, secondary, tertiary = map(self.get_contours, self.get_masks())
        masks = np.dstack([primary, secondary, tertiary])

        height, width, _ = masks.shape

        left, right, top, bottom = bounding_box
        center_x = (right + left) // 2
        center_y = (bottom + top) // 2

        left = max(0, center_x - padding)
        right = min(width, center_x + padding)
        top = max(0, center_y - padding)
        bottom = min(height, center_y + padding)

        return masks[top:bottom, left:right, :]

    def get_image(self):

        """Returns tuple of image channels from cellh5 file"""
        img = self.properties['image/channel/'].value.reshape((3, 1040, 1392))

        """The following were derived by averaging the 99.9 percentiles of
        field 1 images in the wells of dataset 0, concatenating corresponding
        images from the two cell lines.
        """

        # dapi = self.normalise_channel(img[0], 137.1) #199.5)
        # cy5 = self.normalise_channel(img[1], 8.975) #202.65)
        # cy3 = self.normalise_channel(img[2], 51.175) #160.975)

        dapi, cy5, cy3 = map(self.normalise_channel, img)
        return np.dstack([cy5, cy3, dapi])

    def get_crops(self, centers, padding=0, rescale_factor=1):

        img = self.get_image()
        crops = []

        for center in centers:
            crop = self.get_image_crop(img, center, padding)
            crop = crop.astype('float32') / 255
            crop = rescale(crop, rescale_factor, mode='constant')
            crops.append(crop)
        return crops

    def get_image_crop(self, img, center, padding):
        """Plots d-stack of channels at crop location"""
        height, width, _ = img.shape

        center_y, center_x = center

        center_y = np.clip(center_y, padding, height - padding)
        center_x = np.clip(center_x, padding, width - padding)

        left = center_x - padding
        right = center_x + padding
        top = center_y - padding
        bottom = center_y + padding

        return img[top:bottom, left:right, :]


def plot_confusion_matrix(
    ax, matrix, labels, title='Confusion matrix', cmap='Reds', fontsize=9):

    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])
    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels, rotation='90', fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize, minor=True)
    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')
    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)

    # Plot heat map
    proportions = [1. * row / sum(row) for row in matrix]
    ax.pcolor(np.array(proportions[::-1]), cmap=cmap)

    # Plot counts as text
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            confusion = matrix[::-1][row][col]
            if confusion != 0:
                ax.text(col + 0.5, row + 0.5, int(confusion), fontsize=fontsize,
                    horizontalalignment='center',
                    verticalalignment='center')

    # Add finishing touches
    ax.grid(True, linestyle=':')
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('prediction', fontsize=fontsize)
    ax.set_ylabel('actual', fontsize=fontsize)
    # fig.tight_layout()


def image_crop_training_set(df_data, ch5_folder, padding, rescale):

    all_crops = []

    for _, row in df_data[['well', 'field']].drop_duplicates().iterrows():

        well, field = row
        centers = df_data[(df_data['well'] == well) &
                          (df_data['field'] == field)][['y', 'x']].values

        ic = ImageCropper(row['well'], row['field'], ch5_folder)
        # bbs = ic.get_bounding_box_list()
        crops = ic.get_crops(centers, padding, rescale)
        # remove mishappen crops
        w = h = 2 * padding * rescale
        d = 3
        crops = filter(lambda x : x.shape == (h, w, d), crops)
        all_crops.extend(crops)

    return np.stack(all_crops)


def plot_embedded(ax, embedding, labels):

    sns.set(style='darkgrid')

    idx_dmso = labels == 'Neutral'

    ax.scatter(embedding[idx_dmso, 0],
               embedding[idx_dmso, 1],
               color='grey', label='DMSO', s=60, alpha=0.8, edgecolors='black')

    non_neutral_moas = filter(lambda x : x != 'Neutral', np.unique(labels))

    for i, moa in enumerate(non_neutral_moas):
        idx_label = labels == moa
        ax.scatter(embedding[idx_label, 0], embedding[idx_label, 1],
                   color=palette[i], label=moa, s=30, alpha=0.8,
                   edgecolors='black')


def plot_moa_samples(x_img, df_data, moas, nb_samples=7):

    nb_classes = len(moas)

    fig, axes = plt.subplots(figsize=(nb_classes, nb_samples),
                             ncols=nb_classes, nrows=nb_samples)

    for i, moa in enumerate(moas):
        idx = np.arange(df_data.shape[0])[df_data.moa == moa]
        for j in range(nb_samples):
            axes[j][i].imshow(x_img[np.random.choice(idx)])
            axes[j][i].axis('off')
            if j == 0:
                axes[j][i].set_title(moa[:5])

    plt.show()


def plot_embeddings(x_train, sample_size):

    model = UMAP(n_components=2)
    embedding_tsne = model.fit(x_train).embedding_

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(embedding_tsne[:2*sample_size, 0],
               embedding_tsne[:2*sample_size, 1],
               color='#50e3c2', alpha=0.5,
               edgecolor='dimgrey', s=80)

    ax.scatter(embedding_tsne[2*sample_size:, 0],
               embedding_tsne[2*sample_size:, 1],
               color='#f5a623', alpha=0.5,
               edgecolor='dimgrey', s=80)
