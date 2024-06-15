import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score

def plot_metrics(epochs, train_losses, train_accuracies, val_losses, val_accuracies, directory1, directory2):
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(directory1, dpi=300)
    plt.show()
    
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(directory2, dpi=300)
    plt.show()

def net_eval(val_test, n, eval_loader, device, net, loss, loginf):
    eval_loss = 0
    eval_num = 0
    eval_correct = 0
    eval_start = datetime.now()
    for eval_X, eval_Y in tqdm(eval_loader, total=len(eval_loader)):
        eval_X, eval_Y = eval_X.float().to(device), eval_Y.to(device)
        eval_pred = net(eval_X)
        eval_loss += loss(eval_pred, eval_Y).item()
        eval_num += len(eval_Y)
        _, predicted = eval_pred.max(1)
        eval_correct += predicted.eq(eval_Y).sum().item()
    eval_loss_mean = eval_loss / eval_num
    eval_acc = eval_correct / eval_num * 100
    eval_end = datetime.now()
    eval_time = (eval_end - eval_start).total_seconds()
    loginf('{} num: {} — {} loss: {} — {} accuracy: {} — Time: {}'.format(val_test, eval_num, val_test, eval_loss_mean, val_test, eval_acc, eval_time))
    loginf('_' * n)
    return eval_loss_mean, eval_acc

def evaluate(net, device, testloader, class_names, saving_directory):
    net.eval()
    y_pred_labels, y_pred_probs = [], []
    targets_test = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred_labels.extend(predicted.cpu().numpy())
            y_pred_probs.extend(outputs.cpu().numpy()[:, 1])  # Assuming binary classification
            targets_test.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(targets_test, y_pred_labels)
    precision = precision_score(targets_test, y_pred_labels, average='binary')
    recall = recall_score(targets_test, y_pred_labels, average='binary')
    f1 = f1_score(targets_test, y_pred_labels, average='binary')
    fpr, tpr, thresholds = roc_curve(targets_test, y_pred_probs)
    auc_score = roc_auc_score(targets_test, y_pred_labels) 
    cm = confusion_matrix(targets_test, y_pred_labels)
    tn, fp, fn, tp = confusion_matrix(targets_test, y_pred_labels).ravel()
    specificity = tn / (tn+fp)
    print('TEST METRICS')
    print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {auc_score:.2f}')
    
    metrics_data = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_score,
        'Specificity': specificity
    }
    metrics_df = pd.DataFrame([metrics_data])
    csv_file_path = saving_directory + f'/test_metrics.csv'
    metrics_df.to_csv(csv_file_path, index=False)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color='b',  lw=1, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(saving_directory + '/roc_curve.png', dpi=300)
    plt.show()

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(saving_directory + f'/confusion_matrix.png', dpi=300)
    plt.show()

    return accuracy, precision, recall, f1, auc_score, specificity

def TrainModel(
        net,
        device,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        optimizer,
        loss,
        loginf,
        wandb,
        file_name
):
    train_losses = []  
    train_accuracies = []  
    val_losses = []  
    val_accuracies = []  

    #saving_best = 0
    saving_best = np.inf

    for epoch in range(n_epochs):
        # train
        net.train()

        train_loss = 0
        train_accuracy = 0
        train_num = 0
        t_start = datetime.now()
        for X, Y in tqdm(trainloader, total=len(trainloader)):
            X, Y = X.float().to(device), Y.to(device)
            optimizer.zero_grad()
            pred = net(X)
            batch_loss = loss(pred, Y)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            correct = (pred.argmax(1) == Y).type(torch.float).sum().item()
            train_accuracy += correct
            train_num += len(Y)
        train_loss_mean = train_loss / train_num
        train_accuracy_mean = train_accuracy / train_num  
        train_accuracy_mean = train_accuracy_mean * 100
        train_losses.append(train_loss_mean)
        train_accuracies.append(train_accuracy_mean)
        t_end = datetime.now()
        epoch_time = (t_end - t_start).total_seconds()
        loginf('Epoch: {}'.format(epoch))
        loginf('Train num: {}  — Train loss: {} — Train Acc: {} — Time: {}'.format(train_num, train_loss_mean, train_accuracy_mean, epoch_time))

        # validation and test
        with torch.no_grad():
            net.eval()
            val_loss_mean, val_acc = net_eval('Val', 80, valloader, device, net, loss, loginf)
            val_losses.append(val_loss_mean)
            val_accuracies.append(val_acc)
            if val_loss_mean <= saving_best:
                saving_best = val_loss_mean
                best_acc = val_acc
                torch.save(net.state_dict(), file_name)
                val_loss_mean, test_acc = net_eval('Test', 120, testloader, device, net, loss, loginf)

    if wandb:
        for epoch in range(n_epochs):
            wandb.log({
                "epoch": epoch,
                "train loss": train_losses[epoch],
                "train accuracy": train_accuracies[epoch],
                "val loss": val_losses[epoch],
                "val accuracy": val_accuracies[epoch]
            })

    loginf('Training complete.')
    loginf('Best Validation Accuracy: {}'.format(best_acc))
    loginf('Best Test Accuracy: {}'.format(test_acc))
    loginf('_' * 200)
    return train_losses, train_accuracies, val_losses, val_accuracies
