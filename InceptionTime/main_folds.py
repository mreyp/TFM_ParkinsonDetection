from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES

from utils.utils_folds import read_all_datasets
from utils.utils_folds import transform_labels
from utils.utils_folds import create_directory
from utils.utils_folds import run_length_xps
from utils.utils_folds import generate_results_csv
import pandas as pd

import utils
import numpy as np
import sys
import sklearn
import os
import shutil


def fit_classifier(classifier_name, output_directory, dataset_id, id):
    folds = 5
    for fold_index in range(1, folds+1): 
        """
        Fits a classifier using training, validation, and test data for 10 different folds.
        """
        nb_classes = 2  # Ajustar según tus datos específicos
        fold_files_directory = f'TFM_MartaRey/datos/sets/folds_5/files_{id}/'

        # Cargar datos
        x_train = np.load(os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_{fold_index}.npy'))
        y_train = np.load(os.path.join(fold_files_directory, f'y_train_{dataset_id}_fold_{fold_index}.npy'))
        x_val = np.load(os.path.join(fold_files_directory, f'X_val_{dataset_id}_fold_{fold_index}.npy'))
        y_val = np.load(os.path.join(fold_files_directory, f'y_val_{dataset_id}_fold_{fold_index}.npy'))
        x_test = np.load(os.path.join(fold_files_directory, f'X_test_{dataset_id}_fold_{fold_index}.npy'))
        y_test = np.load(os.path.join(fold_files_directory, f'y_test_{dataset_id}_fold_{fold_index}.npy'))

        nb_classes = len(np.unique(np.concatenate((y_train, y_val, y_test), axis=0)))

        # Transforma las etiquetas a un formato consistente y unificado
        y_train, y_val, y_test = transform_labels(y_train, y_val, y_test)

        y_true_val = y_val.astype(np.int64)
        y_true_test = y_test.astype(np.int64)
        y_true_train = y_train.astype(np.int64)

        # Transformación de labels si es necesario, ejemplo usando OneHotEncoder
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(np.concatenate((y_train, y_val, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        if len(x_train.shape) == 2:  # if univariate
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        input_shape = x_train.shape[1:]

        classifier = create_classifier(classifier_name, input_shape, nb_classes,
                                    output_directory, verbose=True)
        
        classifier.fit(x_train, y_train, x_val, y_val, y_true_val, x_test, y_test, y_true_test, fold_index, plot_test_acc=True)
        print(f'Fold {fold_index} processing complete.')
    
    classifier.aggregate_results(folds, nb_classes)
    print('All folds processed and saved.')
        


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):
    if classifier_name == 'inception':
        from classifiers import inception_folds
        return inception_folds.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                              build=build)

def get_xp_val(xp):
    if xp == 'batch_size':
        xp_arr = [16, 32, 128]
    elif xp == 'use_bottleneck':
        xp_arr = [False]
    elif xp == 'use_residual':
        xp_arr = [False]
    elif xp == 'nb_filters':
        xp_arr = [16, 64]
    elif xp == 'depth':
        xp_arr = [3, 9]
    elif xp == 'kernel_size':
        xp_arr = [8, 64]
    else:
        raise Exception('wrong argument')
    return xp_arr


############################################### main
#root_dir = 'TFM_MartaRey/InceptionTime'
xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
       'kernel_size', 'batch_size']
classifier_name = 'inception'

if len(sys.argv) > 1:
    classifier_name = sys.argv[1]
if len(sys.argv) > 2:
    dataset_id = sys.argv[2]
if len(sys.argv) > 3:
    id = sys.argv[3]
if len(sys.argv) > 4:
    root_dir = sys.argv[4]

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
        print(f"Carpeta '{root_dir}' eliminada.")
    os.makedirs(root_dir)
    print(f"Carpeta '{root_dir}' creada.")

    tmp_output_directory = root_dir + '/results/' + classifier_name + '/dataset_' + dataset_id + '/'

    output_directory = tmp_output_directory  + '/'

    if os.path.exists(output_directory):
        print('Already done')
    else:
        create_directory(output_directory)

        fit_classifier(classifier_name, output_directory, dataset_id, id)
        print('Classification process completed for all folds.')

