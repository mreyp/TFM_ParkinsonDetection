from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import pandas as pd
import sys
import sklearn
import utils
import shutil
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets



def fit_classifier():
    '''
    ### PARKINSON DATASET
    x_train = np.load('TFM_MartaRey/datos/sets/X_train_40_1e6_N.npy')
    y_train = np.load('TFM_MartaRey/datos/sets/y_train_40_1e6_N.npy')
    
    x_val = np.load('TFM_MartaRey/datos/sets/X_val_40_1e6_N.npy')
    y_val = np.load('TFM_MartaRey/datos/sets/y_val_40_1e6_N.npy')
    
    x_test = np.load('TFM_MartaRey/datos/sets/X_test_real_N.npy')
    y_test  = np.load('TFM_MartaRey/datos/sets/y_test_real_N.npy')

    '''
    
    ### WHALES DATASET
    data_train = pd.read_csv('TFM_MartaRey/datos/whale_dataset/RightWhaleCalls_train.csv')
    y_train = np.array(data_train['label'])
    x_train = np.array(data_train.drop(['label'], axis=1))

    data_val = pd.read_csv('TFM_MartaRey/datos/whale_dataset/RightWhaleCalls_val.csv')
    y_val = np.array(data_val['label'])
    x_val = np.array(data_val.drop(['label'], axis=1))

    data_test = pd.read_csv('TFM_MartaRey/datos/whale_dataset/RightWhaleCalls_test.csv')
    y_test = np.array(data_test['label'])
    x_test = np.array(data_test.drop(['label'], axis=1))
    
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true_val = np.argmax(y_val, axis=1)
    y_true_test = np.argmax(y_test, axis=1)
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Ajusta el scaler solo a los datos de entrenamiento para evitar data leakage
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    # Usa el scaler ajustado para transformar los conjuntos de validación y prueba
    x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    '''
    '''
    def normalize_series(series_set):
        scaler = sklearn.preprocessing.StandardScaler()
        series_set_scaled = np.zeros_like(series_set)  # Crear un arreglo del mismo tamaño que series_set
        for i, series in enumerate(series_set):
            series_reshaped = series.reshape(-1, 1)
            scaler.fit(series_reshaped)
            series_set_scaled[i] = scaler.transform(series_reshaped).flatten()  # Aplanar de nuevo a (n_features,)
        return series_set_scaled
    x_train = normalize_series(x_train)
    x_val = normalize_series(x_val)
    x_test = normalize_series(x_test)
    '''
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_val, y_val, y_true_val, x_test, y_test, y_true_test)
    classifier.aggregate_results(nb_classes)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
############################################### main

# change this directory for your machine
#root_dir = 'TFM_MartaRey/ResNet'


if len(sys.argv) > 1:
    classifier_name = 'resnet'
if len(sys.argv) > 2:
    root_dir = sys.argv[2]

if os.path.exists(root_dir):
    shutil.rmtree(root_dir)
    print(f"Carpeta '{root_dir}' eliminada.")
os.makedirs(root_dir)
print(f"Carpeta '{root_dir}' creada.")

output_directory = root_dir + '/results/' + classifier_name +'/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', classifier_name)

if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:

    create_directory(output_directory)

    fit_classifier()

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')