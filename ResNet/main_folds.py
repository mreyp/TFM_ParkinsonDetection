from utils.utils_folds import generate_results_csv
from utils.utils_folds import create_directory
from utils.utils_folds import read_dataset
from utils.utils_folds import transform_mts_to_ucr_format
from utils.utils_folds import visualize_filter
from utils.utils_folds import viz_for_survey_paper
from utils.utils_folds import viz_cam
import os
import numpy as np
import pandas as pd
import sys
import sklearn
import utils
import shutil
import os
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
#from classifiers.resnet_folds import aggregate_metrics_and_confusion_matrix


def fit_classifier(classifier_name, output_directory, dataset_id, id):
    """
    Fits a classifier using training, validation, and test data for 10 different folds.
    """
    nb_classes = 2  # Ajustar según tus datos específicos
    folds = 5

    for fold_index in range(1, folds+1):  # Suponiendo que los folds están numerados del 1 al 10

        # Ruta de la carpeta específica donde se encuentran los archivos de cada fold
        fold_files_directory = f'TFM_MartaRey/datos/sets/folds_5/files_{id}/'

        # Cargar datos
        x_train = np.load(os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_{fold_index}.npy'))
        y_train = np.load(os.path.join(fold_files_directory, f'y_train_{dataset_id}_fold_{fold_index}.npy'))
        x_val = np.load(os.path.join(fold_files_directory, f'X_val_{dataset_id}_fold_{fold_index}.npy'))
        y_val = np.load(os.path.join(fold_files_directory, f'y_val_{dataset_id}_fold_{fold_index}.npy'))
        x_test = np.load(os.path.join(fold_files_directory, f'X_test_{dataset_id}_fold_{fold_index}.npy'))
        y_test = np.load(os.path.join(fold_files_directory, f'y_test_{dataset_id}_fold_{fold_index}.npy'))

        # Transformación de labels si es necesario, ejemplo usando OneHotEncoder
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        if len(x_train.shape) == 2:  # if univariate
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        input_shape = x_train.shape[1:]
        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

        y_true_val = np.argmax(y_val, axis=1)
        y_true_test = np.argmax(y_test, axis=1)

        classifier.fit(x_train, y_train, x_val, y_val, y_true_val, x_test, y_test, y_true_test, fold_index)
        print(f'Fold {fold_index} processing complete.')
    classifier.aggregate_results(folds, nb_classes)

    print('All folds processed and saved.')


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'resnet':
        from classifiers import resnet_folds
        return resnet_folds.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
############################################### main

# change this directory for your machine
classifier_name = 'resnet'  # Valor predeterminado para el clasificador
dataset_id = 'default_dataset_id'  # Valor predeterminado para el ID del dataset
#root_dir = 'TFM_MartaRey/ResNet/'

# Revisa si se han proporcionado argumentos y actualiza según sea necesario
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

    output_directory = root_dir + '/results/' + classifier_name + '/dataset_' + dataset_id + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', classifier_name)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)

        fit_classifier(classifier_name, output_directory, dataset_id, id)
        print('Classification process completed for all folds.')
