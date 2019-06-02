import os
import pandas as pd
import numpy as np
import h5py
from utils import *


wells_to_remove = {

    '25D_20_384_20X_D_F_C3_C5_Decon_20160031_2' : {
        ('A02', '1'), ('A02', '2'), ('A02', '3'), ('A02', '4'),
        ('D22', '1'), ('D22', '2'), ('D22', '3'), ('D22', '4'),
        ('H22', '1'), ('H22', '2'), ('H22', '3'), ('H22', '4'),
        ('P07', '1'), ('P07', '2'), ('P07', '3'), ('P07', '4')},

    '25D_20_384_20X_D_F_C3_C5_Decon_20160032_1' : {
        ('A02', '1'), ('A02', '2'), ('A02', '3'), ('A02', '4'),
        ('D22', '1'), ('D22', '2'), ('D22', '3'), ('D22', '4'),
        ('H22', '1'), ('H22', '2'), ('H22', '3'), ('H22', '4'),
        ('P07', '1'), ('P07', '2'), ('P07', '3'), ('P07', '4')},

    '22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231' : {
        ('A02', '1'), ('A02', '2'), ('A02', '3'), ('A02', '4'),
        ('E15', '1'), ('E15', '4'),
        ('G09', '3'),
        ('G17', '3'),
        ('G24', '1'),
        ('H14', '3'),
        ('H18', '1'),
        ('J10', '1'),
        ('J20', '4'),
        ('L12', '1'),
        ('M05', '4'),
        ('P07', '1')},

    '22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468' : {
        ('A02', '1'), ('A02', '2'), ('A02', '3'), ('A02', '4'),
        ('A07', '1'), ('A07', '2'), ('A07', '3'), ('A07', '4'),
        ('A09', '1'), ('A09', '2'), ('A09', '3'), ('A09', '4'),
        ('A10', '1'), ('A10', '2'), ('A10', '3'), ('A10', '4'),
        ('A13', '1'), ('A13', '2'), ('A13', '3'), ('A13', '4'),
        ('B07', '1'), ('B07', '2'), ('B07', '3'), ('B07', '4'),
        ('B09', '1'), ('B09', '2'), ('B09', '3'), ('B09', '4'),
        ('C01', '4'),
        ('C20', '1'), ('C20', '2'),
        ('D22', '1'),
        ('E17', '4'),
        ('F10', '1'),
        ('F23', '1'),
        ('G06', '1'), ('G06', '2'), ('G06', '3'),
        ('G17', '1'),
        ('H12', '1'), ('H12', '2'),
        ('H22', '4'),
        ('I05', '1'), ('I05', '2'), ('I05', '3'), ('I05', '4'),
        ('J05', '1'), ('J05', '2'), ('J05', '3'), ('J05', '4'),
        ('J19', '2'),
        ('J23', '4'),
        ('K06', '1'), ('K06', '2'), ('K06', '3'), ('K06', '4'),
        ('L05', '1'), ('L05', '2'), ('L05', '3'), ('L05', '4'),
        ('M02', '1'),
        ('M06', '1'), ('M06', '2'), ('M06', '3'), ('M06', '4'),
        ('N06', '1'), ('N06', '2'), ('N06', '3'), ('N06', '4'),
        ('N22', '1'), ('N22', '2'), ('N22', '3'), ('N22', '4'),
        ('O07', '1'), ('O07', '2'), ('O07', '3'), ('O07', '4'),
        ('P08', '1'), ('P08', '2'), ('P08', '3'), ('P08', '4'),
        ('P09', '1'), ('P09', '2'), ('P09', '3'), ('P09', '4'),
        ('P10', '1'), ('P10', '2'), ('P10', '3'), ('P10', '4'),
        ('P12', '1'), ('P12', '2'), ('P12', '3'), ('P12', '4'),
        ('P16', '1'), ('P16', '2'), ('P16', '3'), ('P16', '4'),
        ('P24', '2')},

    '12CellLines-morphology': {}

    }


class FeatureReader(object):

    def __init__(self, plate, ch5_folder, channels, cell_line=''):
        """Initialises FeatureReader

        Args:
            plate(str): The directory name of the plate.
            ch5_folder(str): The root directory of ch5 outputs.
            channels(list(str)): The list of channels for to read from.

        """

        self.plate = plate
        self.ch5_folder = ch5_folder
        self.channels = channels

        # import pdb ; pdb.set_trace()

        self.f = h5py.File(os.path.join(ch5_folder, '_all_positions.ch5'), 'r+')
        self.f_feats = self.f['sample/0/plate/%s/experiment' % self.plate]

        self.wells = [well for well in self.f_feats]
        fields = [pos for pos in self.f_feats['%s/position/' % self.wells[0]]]
        imgs = [(well, field) for well in self.wells for field in fields]
        self.imgs = filter(lambda x : not x in wells_to_remove[self.plate], imgs)
        self.cell_line = cell_line

    def get_ids(self):
        """Returns list of cell ids -- (well, field, local_id)

        Returns:
            list: list of cell id lists

        """

        path = '%s/position/%s/feature/primary__primary4/object_features'
        idx = [[well, pos, i] for well, pos in self.imgs 
            for i in range(self.f_feats[path % (well, pos)].len())]
        return idx

    def get_channel_data(self, channel):
        """Retrieves features from hdf5 outputs for a given channel

        Args:
            channel (str): The channel to retrieve features for.

        Returns:
            np.array: The channel feature data.

        """

        feats = [self.f_feats['%s/position/%s/feature/%s/object_features' % 
            (well, pos, channel)].value for well, pos in self.imgs]
        feats = filter(lambda x: not x.shape == (0, 0), feats)
        return np.vstack(feats)

    def get_feature_names(self, channel):
        """Returns list of feature names corresponding with the numpy array

        Args:
            channel (str): The channel to retrieve features names for.

        Returns:
            list: The list of feature names for channel.

        """

        return [channel + '_' + feature[0] for feature in 
            self.f['definition/feature/%s/object_features' % channel]]

    def get_centers(self, channel):
        """Returns centers of cells on a given channel

        Args:
            channel (str): The channel to retrieve cell center coordinates from.

        Returns:
            list: The x- and y-coordinates of cells on channel.

        """

        centers = [self.f_feats['%s/position/%s/feature/%s/center' % 
            (well, pos, channel)].value for well, pos in self.imgs]
        centers = filter(lambda x: not x.shape == (0, 0), centers)
        return map(list, np.hstack(centers))

    def get_classifications(self, channel):
        """
        """

        labels = [self.f_feats['%s/position/%s/feature/%s/' % 
            (well, pos, channel) + 'object_classification/prediction'].value 
            for well, pos in self.imgs]
        return [label[0] for label in np.hstack(labels)]

    def join_plate_map(self, df_metadata):
        """Joins metadata to self.data (feature dataframe)

        Args:
            pandas.DataFrame: plate metedata.

        """

        self.data = self.data.join(df_metadata.set_index('well'), on='well')

    def read_all(self):
        """Compiles all data into a pandas DataFrame

        Returns:
            pandas.DataFrame: All extracted features and centers

        """

        # initialise DataFrame with ids
        print('Reading cell data...')
        idx = self.get_ids()
        self.data = pd.DataFrame(idx, columns=['well', 'field', 'cell_index'])
        self.feature_names = []

        # add centers
        print('Reading cell centers...')
        centers = self.get_centers(self.channels[0])
        df_centers = pd.DataFrame(centers, columns=['x', 'y'])

        self.data = pd.concat([self.data, df_centers], axis=1)

        # add channel data
        for channel in self.channels:
            print('Reading %s features...' % channel)
            # read channel data
            channel_data = self.get_channel_data(channel)
            feature_names = self.get_feature_names(channel)
            self.feature_names.extend(feature_names)
            df_channel_data = pd.DataFrame(channel_data, columns=feature_names)
            # append to main data frame
            self.data = pd.concat([self.data, df_channel_data], axis=1)

        # remove NaNs
        print('Removing invalid data...')
        self.data = self.data.dropna(axis=0, how='any')
        # assign cell line
        self.data['cell_line'] = self.cell_line

        print('Done!')

    def save_data(self):
        """Saves data frame to /tmp directory

        """

        print 'Saving data...'
        self.data.to_pickle('/tmp/%s.pkl' % self.plate)
        print 'Done!'
