import os
import sys
import wandb
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR 
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from time_config import config
from time_train_folds import TrainModel
from time_train_folds import plot_metrics
from time_train_folds import evaluate


sys.path.append('../')
from Models.net_conv import CONV
from Models.net_conv_rf import receptive_field
from Models.net_rnn import RNN
from Models.net_xformer import Xformer
from Models.utils import seed_everything, DatasetCreator

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--task', type=str, default='RightWhaleCalls')
# parser.add_argument('--task', type=str, default='FruitFlies')
# parser.add_argument('--task', type=str, default='MosquitoSound')
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--saving', type=str, default='TFM_MartaRey/CDIL_CNN/results')
args = parser.parse_args()

SAVING = args.saving

if os.path.exists(SAVING):
    shutil.rmtree(SAVING)
    print(f"Carpeta '{SAVING}' eliminada.")
os.makedirs(SAVING)
print(f"Carpeta '{SAVING}' creada.")

all_predictions = []
all_true_labels = []
all_metrics = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)
all_folds_results = []
folds = 5

for fold_index in range(1, folds+1):  # Assuming folds are numbered from 1 to 10

    print('Training  on fold:', fold_index)


    # Config
    use_wandb = False
    INPUT_SIZE = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TASK = args.task
    MODEL = args.model
    SEED = args.seed

    seed_everything(args.seed)
    if use_wandb:
        wandb.init(project=TASK, name=MODEL + str(SEED), entity="leic-no")
        WANDB = wandb
    else:
        WANDB = None

    cfg_training = config[TASK]['training']
    cfg_model = config[TASK]['models']

    BATCH = cfg_training['batch_size']

    # Load data for current fold
    data_directory = f'TFM_MartaRey/datos/sets/folds_5/files_real_40_1e5_N/'
    dataset_id = '40_1e5_N'
    id = '1'
    features_train = np.load(os.path.join(data_directory, f'X_train_{dataset_id}_fold_{fold_index}.npy'))
    labels_train = np.load(os.path.join(data_directory, f'y_train_{dataset_id}_fold_{fold_index}.npy'))
    features_val = np.load(os.path.join(data_directory, f'X_val_{dataset_id}_fold_{fold_index}.npy'))
    labels_val = np.load(os.path.join(data_directory, f'y_val_{dataset_id}_fold_{fold_index}.npy'))
    features_test = np.load(os.path.join(data_directory, f'X_test_{dataset_id}_fold_{fold_index}.npy'))
    labels_test = np.load(os.path.join(data_directory, f'y_test_{dataset_id}_fold_{fold_index}.npy'))

    # Convert to tensors
    features_train = torch.tensor(features_train).view(len(labels_train), -1, 1).float()
    labels_train = torch.tensor(labels_train)
    features_val = torch.tensor(features_val).view(len(labels_val), -1, 1).float()
    labels_val = torch.tensor(labels_val)
    features_test = torch.tensor(features_test).view(len(labels_test), -1, 1).float()
    labels_test = torch.tensor(labels_test)

    # Create datasets and loaders
    trainset = DatasetCreator(features_train, labels_train)
    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, drop_last=False)
    valset = DatasetCreator(features_val, labels_val)
    valloader = DataLoader(valset, batch_size=BATCH, shuffle=False, drop_last=False)
    testset = DatasetCreator(features_test, labels_test)
    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False, drop_last=False)


    CLASS = len(torch.unique(labels_train))
    SEQ_LEN = features_train.shape[1]
    LAYER = int(np.log2(SEQ_LEN)) 


    # Model
    if MODEL == 'CDIL' or MODEL == 'DIL' or MODEL == 'TCN' or MODEL == 'CNN':
        NHID = cfg_model['cnn_hidden']
        KERNEL_SIZE = cfg_model['cnn_ks']
        net = CONV(TASK, MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE)
        receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
    elif MODEL == 'Deformable':
        NHID = cfg_model['cnn_hidden']
        KERNEL_SIZE = cfg_model['cnn_ks']
        net = CONV(TASK, 'CNN', INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, True)
        receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
    elif MODEL == 'LSTM' or MODEL == 'GRU':
        LAYER = cfg_model['rnn_layer']
        NHID = cfg_model['rnn_hidden']
        net = RNN(TASK, MODEL, INPUT_SIZE, CLASS, NHID, LAYER)
    elif MODEL == 'Transformer' or MODEL == 'Linformer' or MODEL == 'Performer':
        DIM = cfg_model['former_dim']
        DEPTH = cfg_model['former_depth']
        HEADS = cfg_model['former_head']
        net = Xformer(MODEL, INPUT_SIZE, CLASS, DIM, SEQ_LEN, DEPTH, HEADS)
    else:
        print('no this model.')
        sys.exit()

    net = net.to(device)
    para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)


    # Log
    if MODEL == 'Transformer' or MODEL == 'Linformer' or MODEL == 'Performer':
        file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_H' + str(DIM)
    else:
        file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)

    os.makedirs('TFM_MartaRey/CDIL_CNN/experiments/_3time/time_log/', exist_ok=True)
    os.makedirs('TFM_MartaRey/CDIL_CNN/experiments/_3time/time_model/', exist_ok=True)
    log_file_name = 'TFM_MartaRey/CDIL_CNN/experiments/_3time/time_log/' + id + str(fold_index) + file_name + '.txt'
    model_name = 'TFM_MartaRey/CDIL_CNN/experiments/_3time/time_model/' + id + str(fold_index) + file_name + '.ph'
    handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
    loginf = logging.info

    loginf(torch.cuda.get_device_name(device))
    loginf(file_name)

    # Optimize
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    # train
    train_losses, train_accuracies, val_losses, val_accuracies = TrainModel(
        net=net,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        n_epochs=cfg_training['epoch'],
        optimizer=optimizer,
        loss=loss,
        loginf=loginf,
        wandb=WANDB,
        file_name=model_name
    )

    directory1 = SAVING + f'/epochs_loss{fold_index}.png'
    directory2 = SAVING + f'/epochs_accuracy{fold_index}.png'
    class_names=['Control', 'Parkinson']

    net.load_state_dict(torch.load(model_name))
    
    plot_metrics(cfg_training['epoch'], train_losses, train_accuracies, val_losses, val_accuracies, directory1, directory2)

    # Evaluate the model
    accuracy, precision, recall, f1, auc_score, specificity, fpr, tpr, thresholds, targets_test, y_pred_labels = evaluate(net, device, testloader, class_names,  SAVING, fold_index, loss)

    all_predictions.extend(y_pred_labels)
    all_true_labels.extend(targets_test)
    all_metrics.append([accuracy, precision, recall, f1, auc_score, specificity])

    # Interpolation of TPR for common FPR
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

mean_metrics = np.mean(all_metrics, axis=0)
std_metrics= np.std(all_metrics, axis=0)
# Pearson Variation Coefficient
cv_metrics = (std_metrics / mean_metrics)

print(f"Mean Metrics - Accuracy: {mean_metrics[0]:.2f}, Precision: {mean_metrics[1]:.2f}, Recall: {mean_metrics[2]:.2f}, F1 Score: {mean_metrics[3]:.2f}, AUC: {mean_metrics[4]:.2f}, Specificty: {mean_metrics[5]:.2f}")
print(f"Variation Coefficients - Accuracy: {cv_metrics[0]:.2f}, Precision: {cv_metrics[1]:.2f}, Recall: {cv_metrics[2]:.2f}, F1 Score: {cv_metrics[3]:.2f}, AUC: {cv_metrics[4]:.2f}, Specificty CV: {cv_metrics[5]:.2f}")

data = {
    'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity'],
    'Values': mean_metrics,
    'CV Metrics': ['Accuracy CV', 'Precision CV', 'Recall CV', 'F1 Score CV', 'AUC CV', 'Specificity CV'],
    'CV Values': cv_metrics
}

metrics_df = pd.DataFrame(data)
metrics_df = metrics_df.T
metrics_df.columns = metrics_df.iloc[0]
metrics_df = metrics_df.drop(metrics_df.index[0])
csv_file_path = SAVING + f'/mean_test_metrics.csv'
metrics_df.to_csv(csv_file_path, index=False)
##-----------------------------------------------------------------CONFUSION MATRIX

cm_all = confusion_matrix(all_true_labels, all_predictions)

print(classification_report(all_true_labels, all_predictions, target_names=class_names))

plt.figure(figsize=(5,4))
sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Accumulated Confusion Matrix')
plt.savefig(SAVING + '/accumulated_confusion_matrix.png', dpi=300)
plt.show()

##-----------------------------------------------------------------ROC CURVE
### auc_all vs mean_auc

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
plt.savefig(SAVING + '/mean_roc_curve.png', dpi=300)
plt.show()