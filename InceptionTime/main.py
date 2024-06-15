from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES

from utils.utils import read_all_datasets
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import run_length_xps
from utils.utils import generate_results_csv
import pandas as pd

import utils
import numpy as np
import sys
import sklearn
import os
import shutil


def prepare_data():
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

    nb_classes = len(np.unique(np.concatenate((y_train, y_val, y_test), axis=0)))

    # Transforma las etiquetas a un formato consistente y unificado
    y_train, y_val, y_test = transform_labels(y_train, y_val, y_test)

    y_true_val = y_val.astype(np.int64)
    y_true_test = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # Codifica las etiquetas usando one-hot encoding
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_val, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

 
    # Normalización estándar de los datos
    #scaler = sklearn.preprocessing.MinMaxScaler()
    # Ajusta el scaler solo a los datos de entrenamiento para evitar data leakage
    #x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    # Usa el scaler ajustado para transformar los conjuntos de validación y prueba
    #x_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    #x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    
    def normalize_series(series_set):
        scaler = sklearn.preprocessing.MinMaxScaler()
        series_set_scaled = np.zeros_like(series_set)  # Crear un arreglo del mismo tamaño que series_set
        for i, series in enumerate(series_set):
            series_reshaped = series.reshape(-1, 1)
            scaler.fit(series_reshaped)
            series_set_scaled[i] = scaler.transform(series_reshaped).flatten()  # Aplanar de nuevo a (n_features,)
        return series_set_scaled
    #x_train = normalize_series(x_train)
    #x_val = normalize_series(x_val)
    #x_test = normalize_series(x_test)

    # Ajusta la forma de los conjuntos de datos si son univariados
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
    return x_train, y_train, x_val, y_val, x_test, y_test,  y_true_val, y_true_test, nb_classes, y_true_train, enc 


def fit_classifier():
    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes,
                                   output_directory, verbose=True)
    if classifier_name == 'inception':
        classifier.fit(x_train, y_train, x_val, y_val, y_true_val, plot_test_acc=True)

    if classifier_name == 'nne':
        classifier.fit(x_val, y_val, y_true_val, x_test, y_test, y_true_test)
        classifier.aggregate_results()
def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
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
    root_dir = sys.argv[2]
    nb_iter_ = 1

if os.path.exists(root_dir):
    shutil.rmtree(root_dir)
    print(f"Carpeta '{root_dir}' eliminada.")
os.makedirs(root_dir)
print(f"Carpeta '{root_dir}' creada.")

    #datasets_dict = read_all_datasets(root_dir, archive_name)

for iter in range(nb_iter_):
    print('\t\titer', iter)

    trr = ''
    if iter != 0:
        trr = '_itr_' + str(iter)

    tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + trr 

    x_train, y_train, x_val, y_val, x_test, y_test, y_true_val, y_true_test, nb_classes, y_true_train, enc = prepare_data()

    output_directory = tmp_output_directory  + '/'

    temp_output_directory = create_directory(output_directory)

    if temp_output_directory is None:
        print('Already_done', tmp_output_directory)
        continue

    fit_classifier()

    print('\t\t\t\tDONE')

# run the ensembling of these iterations of Inception
classifier_name = 'nne'

tmp_output_directory_nne = root_dir + '/results/' + classifier_name 

x_train, y_train, x_val, y_val, x_test, y_test, y_true_val, y_true_test, nb_classes, y_true_train, enc = prepare_data()

output_directory = tmp_output_directory_nne + '/'

fit_classifier()

print('\t\t\t\tDONE')

