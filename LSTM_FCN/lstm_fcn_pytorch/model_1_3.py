### NORMALIZACION MIN-MAX INDEPENDIENTE

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from lstm_fcn_pytorch.modules import LSTM_FCN
import math
import pandas as pd

class Model():
    def __init__(self, x, y, units, dropout, dropout2, filters, kernel_sizes, val_data):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        x, y = shuffle(x, y)
        
        self.weight = compute_class_weight(class_weight="balanced", classes=np.sort(np.unique(y)), y=y)
        
        model = LSTM_FCN(
            input_length=x.shape[1],
            units=units,
            dropout=dropout,
            dropout2=dropout2,
            filters=filters,
            kernel_sizes=kernel_sizes,
            num_classes=len(np.unique(y))
        )
        
        self.x = torch.from_numpy(x).to(self.device).float()
        self.y = torch.from_numpy(y).to(self.device).long()
        self.model = model.to(self.device)

        x_val, y_val = val_data
        self.x_val = torch.from_numpy(x_val).to(self.device).float()
        self.y_val = torch.from_numpy(y_val).to(self.device).long()
        
    
    def fit(self, learning_rate, batch_size, epochs, saving_directory, verbose=True):
        
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=batch_size,
            shuffle=True
        )
        val_dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x_val, self.y_val),
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 30, factor = 0.5, min_lr = 1e-7)
        #scheduler = StepLR(optimizer, 30, gamma=0.4)

        '''
        OPTIONS:
            * outputlayer = Linear
            * criterion = nn.CrossEntropyLoss()
        else:
            * outputlayer = Softmax
            * criterion = nn.NLLLoss()
        '''
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weight, dtype=torch.float, device=self.device))

        epoch_losses = []
        epoch_accuracies = []
        epoch_val_losses = []
        epoch_val_accuracies = []
        best_val_loss = np.inf
        best_val_acc = 0
        patience_counter = 0
        patience = 50
        actual_epochs = 0 
        best_model_path = 'best_model.pth' 

        for epoch in range(epochs):
            self.model.train()
            batch_losses = []
            batch_accuracies = []
            for inputs, targets in dataset:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                accuracy = (torch.argmax(outputs, dim=-1) == targets).float().mean()
                batch_losses.append(loss.item())
                batch_accuracies.append(accuracy.item())

            epoch_losses.append(np.mean(batch_losses))
            epoch_accuracies.append(np.mean(batch_accuracies))
            
            # Internal validation
            self.model.eval()
            val_batch_losses = []
            val_batch_accuracies = []
            with torch.no_grad():
                for val_inputs, val_targets in val_dataset:
                    val_outputs = self.model(val_inputs)
                    val_loss = loss_fn(val_outputs, val_targets)
                    val_accuracy = (torch.argmax(val_outputs, dim=-1) == val_targets).float().mean()
                    val_batch_losses.append(val_loss.item())
                    val_batch_accuracies.append(val_accuracy.item())

            epoch_val_losses.append(np.mean(val_batch_losses))
            epoch_val_accuracies.append(np.mean(val_batch_accuracies))
            actual_epochs = epoch + 1 
            current_val_loss = np.mean(val_batch_losses)
            current_val_accuracy = np.mean(val_batch_accuracies)
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(current_val_loss)  # Adjust learning rate based on validation loss
            #scheduler.step()

            # Después de llamar al scheduler, compara la lr para ver si ha cambiado
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != prev_lr and verbose:
                print(f'Learning rate reduced from {prev_lr} to {current_lr} at epoch {epoch+1}')

            if verbose:
                print(f'Epoch {epoch + 1}, Train Loss: {epoch_losses[-1]:.4f}, Train Accuracy: {epoch_accuracies[-1]:.4f}, Val Loss: {epoch_val_losses[-1]:.4f}, Val Accuracy: {epoch_val_accuracies[-1]:.4f}')

            
            if  best_val_loss > current_val_loss :
                best_val_loss = current_val_loss
                patience_counter = 0
                best_train_acc = epoch_accuracies[-1]
                best_val_acc = epoch_val_accuracies[-1]
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), best_model_path)  # Guarda el mejor modelo.
                print(f"Saved best model with validation loss: {best_val_loss} \n")

            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break  # Detiene el entrenamiento
  
        if verbose:
            directory1 = saving_directory + f'/epochs_loss.png'
            directory2 = saving_directory + f'/epochs_accuracy.png'
            self.plot_metrics(actual_epochs, epoch_losses, epoch_accuracies, directory1, directory2, epoch_val_losses, epoch_val_accuracies)
            print((f'Best epoch: {best_epoch},  Train Accuracy: {best_train_acc:.4f}, Val Accuracy: {best_val_acc:.4f}')
)

    def plot_metrics(self, epochs, train_losses, train_accuracies, directory1, directory2, val_losses=None, val_accuracies=None):
        plt.figure(figsize=(5, 4))
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        if val_losses is not None:
            plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(directory1, dpi=300)
        plt.show()
        
        plt.figure(figsize=(5, 4))
        plt.plot(range(1, epochs + 1), train_accuracies, label='Train')
        if val_accuracies is not None:
            plt.plot(range(1, epochs + 1), val_accuracies, label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig(directory2, dpi=300)
        plt.show()

    def load_model(self, model_path):

        # Carga el estado del modelo.
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Cambia el modelo a modo de evaluación.
        
        # Mueve el modelo al dispositivo adecuado.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, x_test):
        
        '''
        Predict the class labels.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, length) where samples is the number of time series
            and length is the length of the time series.
            
        Returns:
        __________________________________
        y_pred: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
        '''
        
        # Scale the data.
        x_tensor = torch.from_numpy(x_test).to(self.device).float()

        # Get the predicted probabilities.
        with torch.no_grad():  
            p = torch.nn.functional.softmax(self.model(x_tensor), dim=-1)
            #p = self.model(x_tensor)
        y_pred = p.cpu().numpy()[:, 1]
        # Convert the output probabilities to predicted class labels.
        y_pred_labels = np.argmax(p.cpu().numpy(), axis=-1)

        return  y_pred_labels, y_pred
    
    def evaluate(self, y_pred_labels, y_pred, targets_test, saving_directory, class_names):
        accuracy = accuracy_score(targets_test, y_pred_labels)
        precision = precision_score(targets_test, y_pred_labels, average = 'binary')
        recall = recall_score(targets_test, y_pred_labels , average = 'binary')
        f1 = f1_score(targets_test, y_pred_labels, average = 'binary')
        cm = confusion_matrix(targets_test, y_pred_labels)
        fpr, tpr, thresholds = roc_curve(targets_test, y_pred)
        auc_score = roc_auc_score(targets_test, y_pred_labels) 

        print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {auc_score:.2f}')
        
        metrics_data = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc_score
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
        return accuracy, precision, recall, f1, auc_score, fpr, tpr, thresholds