# resnet model 
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
from utils.utils_folds import save_test_duration

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils_folds import save_logs
from utils.utils_folds import calculate_metrics
from utils.utils_folds import plot_and_save_confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve



class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        #conv_z = keras.layers.Dropout(0.5)(conv_z) ###############

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)


        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        #conv_z = keras.layers.Dropout(0.5)(conv_z) ####################

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        #conv_z = keras.layers.Dropout(0.5)(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                      metrics=['accuracy'])
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30,
                                                      verbose=True, min_lr=1e-8, mode="min")
        
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, 
                                                       verbose=True, restore_best_weights=True, mode="min")

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True, verbose=True, mode='min')
        
        self.callbacks = [reduce_lr, model_checkpoint, early_stopping]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true_val, x_test, y_test, y_true_test, fold_index):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = 500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + f'last_model{fold_index}.hdf5')

        y_pred_val_proba = self.predict(x_val, y_true_val, return_df_metrics=False)
        y_pred_test_proba = self.predict(x_test, y_true_test, return_df_metrics=False)
        # save predictions
        np.save(self.output_directory + f'y_pred_val{fold_index}.npy', y_pred_val_proba)
        np.save(self.output_directory + f'y_pred_test{fold_index}.npy', y_pred_test_proba)
        np.save(self.output_directory + f'y_true_val{fold_index}.npy', y_true_val)
        np.save(self.output_directory + f'y_true_test{fold_index}.npy', y_true_test)

        # convert the predicted from binary to integer
        y_pred_val = np.argmax(y_pred_val_proba, axis=1)
        y_pred_test = np.argmax(y_pred_test_proba, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred_val, y_pred_val_proba, y_true_val, duration, fold_index)
        df_metrics_test = save_logs(self.output_directory, hist, y_pred_test, y_pred_test_proba, y_true_test, duration, fold_index)

        metrics_val = calculate_metrics(y_true_val, y_pred_val, y_pred_val_proba, duration)
        metrics_test = calculate_metrics(y_true_test, y_pred_test, y_pred_test_proba, duration)
        print('VALIDATION METRICS: \n', metrics_val)
        print('TEST METRICS: \n', metrics_test)
 
        metrics_val.to_csv(self.output_directory + f'df_metrics_val{fold_index}.csv', index=False)
        metrics_test.to_csv(self.output_directory + f'df_metrics_test{fold_index}.csv', index=False)
        class_names = ['Control', 'Parkinson']
        plot_and_save_confusion_matrix(y_true_val, y_pred_val, self.output_directory + f'cm_validation{fold_index}.png', class_names)
        plot_and_save_confusion_matrix(y_true_test, y_pred_test, self.output_directory + f'cm_test{fold_index}.png', class_names)


        keras.backend.clear_session()

        return df_metrics, df_metrics_test

    def predict(self, x_test, y_true, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred_test_proba = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred_test_proba, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, y_pred_test_proba, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred_test_proba
    
    def aggregate_results(self, num_folds, nb_classes):
        metrics = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        total_conf_matrix = np.zeros((nb_classes, nb_classes))

        for fold_index in range(1, num_folds + 1):
            y_pred_test_proba = np.load(self.output_directory + f'y_pred_test{fold_index}.npy')
            y_true_test = np.load(self.output_directory + f'y_true_test{fold_index}.npy')
            y_pred_test = np.argmax(y_pred_test_proba, axis=-1)

            conf_matrix = confusion_matrix(y_true_test, y_pred_test)
            total_conf_matrix += conf_matrix

            fold_metrics = calculate_metrics(y_true_test, y_pred_test, y_pred_test_proba, 0)
            metrics.append(fold_metrics)
            
            fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_test_proba[:, 1])
            # Interpolation of TPR for common FPR
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

             # Calcula las métricas medias
        df_metrics = pd.concat(metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std()
        cv_metrics = (std_metrics / mean_metrics) 

        # Preparing DataFrame in the requested format
        data = {
            'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
            'Values': mean_metrics.values.tolist()[:-1],
            'CV Metrics': ['Accuracy CV', 'Precision CV', 'Recall CV', 'F1 Score CV', 'AUC CV'],
            'CV Values': cv_metrics.values.tolist()[:-1]
        }

        metrics_df = pd.DataFrame(data)
        metrics_df = metrics_df.T
        metrics_df.columns = metrics_df.iloc[0]
        metrics_df = metrics_df.drop(metrics_df.index[0])
        csv_file_path = self.output_directory + '/mean_test_metrics.csv'
        metrics_df.to_csv(csv_file_path, index=False)

        class_names = ['Control', 'Parkinson']

        plt.figure(figsize=(5, 4))
        sns.heatmap(total_conf_matrix, annot=True, fmt="g", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
        plt.title('Accumulated Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(self.output_directory + '/accumulated_confusion_matrix.png', dpi=300)
        plt.show()

        # Calcular la media y la desviación estándar de los TPRs
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(mean_fpr, mean_tpr, color='b',
                label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_metrics[4], std_metrics[4]),
                lw=1)
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        ax.grid()

        # Añadir la variabilidad de la curva ROC (desviación estándar)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                        label='$\pm$ 1 std. dev.')

        # Añadir detalles finales al gráfico
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Mean ROC Curve')
        ax.legend(loc='lower right')
        plt.savefig(self.output_directory  + '/mean_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()


