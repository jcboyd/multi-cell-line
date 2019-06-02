import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def construct_profiles(df_dataset, features, method, cell_count=False):

    """ Creates profile for given methodology

    Args:
        df_dataset(pd.DataFrame): the dataset
        features(list): list of features names
        method(string): the method

    Returns:
        pd.DataFrame: the phenotypic profiles for each drug

    """

    # extract unique wells
    wells = set(df_dataset.well)

    # a matrix of drugs vs. feature indices
    df_profiles = pd.DataFrame(index=wells, columns=features)

    # negative control
    df_dmso = df_dataset[df_dataset['content'] == 'DMSO']

    for well in wells:

        # collect cells from current well
        df_drug = df_dataset[df_dataset['well'] == well]

        if method == 'adams':  # average

            profile = list(df_drug[features].mean(axis=0))
            df_profiles.loc[well][features] = profile

        elif method == 'perlman':  # ks statistic

            profile = [ks_2samp(df_drug[feature], df_dmso[feature])[0]
                       for feature in features]
            df_profiles.loc[well][features] = profile

        elif method == 'loo':  # SVM

            # collect perturbation features
            x_drug = df_drug[features]

            # split random subset
            x_dmso = df_dmso[features]
            _, x_dmso = train_test_split(x_dmso, test_size=x_drug.shape[0])

            # concatenate data
            x_train = np.concatenate([x_dmso, x_drug])
            y_train = np.concatenate([np.zeros(x_dmso.shape[0]),
                                      np.ones(x_drug.shape[0])])

            # train linear SVM
            clf = SVC(kernel='linear')
            clf.fit(x_train, y_train)

            # exract weight vector (normal to hyperplane)
            profile = clf.coef_.T[:, 0]
            df_profiles.loc[well][features] = profile

    if cell_count:
        cell_counts = fr.data.well.value_counts()
        df_profiles = df_profiles.join(cell_counts).rename(columns={'well' : 'cell count'})

    return df_profiles


def lococv(df_profiles, df_metadata, moas, model=KNeighborsClassifier(n_neighbors=1)):

    """ Leave-one-out cross-validation

        Args:
            df_profile(pd.DataFrame): 
            df_metadata(pd.DataFrame):
        Returns:
            (np.ndarray): confusion matrix
    """

    df_profiles = df_profiles.div(np.sum(df_profiles, axis=1), axis=0)
    df_probs = pd.DataFrame(index=df_profiles.index, columns=moas)
    confusion_matrix = np.zeros((moas.shape[0], moas.shape[0]))

    for well, df_holdout in df_profiles.iterrows():
        # hold out well
        holdout_well = df_holdout.name

        # training set (all other profiles)
        df_train = df_profiles[~(df_profiles.index == holdout_well)]

        labels = df_train.join(df_metadata).moa
        model.fit(df_train, labels)

        soft_pred = model.predict_proba(df_holdout.values.reshape(1, -1))
        pred_moa = model.predict(df_holdout.values.reshape(1, -1))
        true_moa = df_metadata.loc[holdout_well].moa

        df_probs[df_probs.index == well] = soft_pred

        # record accuracy
        confusion_matrix[np.where(moas == true_moa)[0][0],
                         np.where(moas == pred_moa)[0][0]] += 1

    return df_probs, confusion_matrix
