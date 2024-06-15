import keras
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import calculate_metrics
from utils.utils import plot_and_save_confusion_matrix
from utils.utils import create_directory
from utils.utils import check_if_file_exists
import gc
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, roc_auc_score


class Classifier_NNE:

    def create_classifier(self, model_name, input_shape, nb_classes, output_directory, verbose=False,
                          build=True):
        if self.check_if_match('inception*', model_name):
            from classifiers import inception
            return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                                  build=build)

    def check_if_match(self, rex, name2):
        import re
        pattern = re.compile(rex)
        return pattern.match(name2)

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, nb_iterations=5,
                 clf_name='inception'):
        self.classifiers = [clf_name]
        out_add = ''
        for cc in self.classifiers:
            out_add = out_add + cc + '-'
        self.archive_name = ARCHIVE_NAMES[0]
        self.iterations_to_take = [i for i in range(nb_iterations)]
        for cc in self.iterations_to_take:
            out_add = out_add + str(cc) + '-'
        self.output_directory = output_directory.replace('nne',
                                                         'nne' + '/' + out_add)
        create_directory(self.output_directory)
        self.dataset_name = output_directory.split('/')[-2]
        self.verbose = verbose
        self.models_dir = output_directory.replace('nne', 'classifier')

    def fit(self, x_val, y_val, y_true_val, x_test, y_test, y_true_test):
        # no training since models are pre-trained
        start_time = time.time()

        y_pred_val_proba = np.zeros(shape=y_val.shape)
        y_pred_test_proba = np.zeros(shape=y_test.shape)

        ll = 0

        # loop through all classifiers
        for model_name in self.classifiers:
            # loop through different initialization of classifiers
            for itr in self.iterations_to_take:
                if itr == 0:
                    itr_str = ''
                else:
                    itr_str = '_itr_' + str(itr)

                curr_archive_name = self.archive_name + itr_str

                curr_dir = self.models_dir.replace('classifier', model_name).replace(
                    self.archive_name, curr_archive_name)

                model = self.create_classifier(model_name, None, None,
                                               curr_dir, build=False)

                predictions_file_name_val = curr_dir + 'y_pred_val.npy'
                predictions_file_name_test = curr_dir + 'y_pred_test.npy'

                # check if predictions already made
                if check_if_file_exists(predictions_file_name_val):
                    curr_y_pred_val = np.load(predictions_file_name_val)
                else:
                    curr_y_pred_val = model.predict(x_val, y_true_val, return_df_metrics=False)
                    np.save(predictions_file_name_val, curr_y_pred_val)
                    keras.backend.clear_session()

                if check_if_file_exists(predictions_file_name_test):
                    # Si existe, carga las predicciones del archivo
                    curr_y_pred_test = np.load(predictions_file_name_test)
                else:
                    curr_y_pred_test = model.predict(x_test, y_true_test, return_df_metrics=False)
                    np.save(predictions_file_name_test, curr_y_pred_test)
                    keras.backend.clear_session()

                y_pred_val_proba = y_pred_val_proba + curr_y_pred_val
                y_pred_test_proba = y_pred_test_proba + curr_y_pred_test

                ll += 1

        # average predictions
        y_pred_val_proba = y_pred_val_proba/ ll
        y_pred_test_proba = y_pred_test_proba / ll

        # save predictions
        np.save(self.output_directory + 'y_pred_val.npy', y_pred_val_proba)
        np.save(self.output_directory + 'y_pred_test.npy', y_pred_test_proba)
        np.save(self.output_directory + 'y_true_val.npy', y_true_val)
        np.save(self.output_directory + 'y_true_test.npy', y_true_test)

        # convert the predicted from binary to integer
        y_pred_val = np.argmax(y_pred_val_proba, axis=1)
        y_pred_test = np.argmax(y_pred_test_proba, axis=1)

        duration = time.time() - start_time

        df_metrics_val = calculate_metrics(y_true_val, y_pred_val, y_pred_val_proba, duration)
        df_metrics_test = calculate_metrics(y_true_test, y_pred_test, y_pred_test_proba, duration)

        print('VALIDATION METRICS: \n', df_metrics_val)
        print('TEST METRICS: \n', df_metrics_test)

        df_metrics_val.to_csv(self.output_directory + 'df_metrics_val.csv', index=False)
        df_metrics_test.to_csv(self.output_directory + 'df_metrics_test.csv', index=False)
        class_names = ['Control', 'Parkinson']
        plot_and_save_confusion_matrix(y_true_val, y_pred_val, self.output_directory + 'cm_validation.png', class_names)
        plot_and_save_confusion_matrix(y_true_test, y_pred_test, self.output_directory + 'cm_test.png', class_names)

        gc.collect()
    
    def aggregate_results(self):

        y_pred_test_proba = np.load(self.output_directory + 'y_pred_test.npy')
        y_true_test = np.load(self.output_directory + 'y_true_test.npy')
        y_pred_test = np.argmax(y_pred_test_proba, axis=-1)

        fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_test_proba[:, 1])
        auc_score = roc_auc_score(y_true_test, y_pred_test)

        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, color='b',  lw=1,label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        plt.grid()
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(self.output_directory  + 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
