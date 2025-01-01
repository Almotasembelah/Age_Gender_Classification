import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, StepLR
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import time
import gc
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate
from IPython.display import clear_output, display, HTML

class ModelManager:
    """
    A utility class for training, validating, and managing PyTorch models.
    Supports single-task and multi-task learning, learning rate scheduling,
    checkpointing, and visualization of training progress.
    """

    def __init__(self, model, optimizer, loss_fn, device=None):
        """
        Initializes the ModelManager instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            loss_fn (torch.nn.Module or list): The loss function(s). For multi-task learning, pass a list of loss functions.
            device (str, optional): The device to use ('cuda' or 'cpu'). Defaults to GPU if available.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print('cuda is not available. Device will be set to `cpu`')
                self.device = 'cpu'

        self.model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self.model.to(self.device)
        self.multi_task = True if isinstance(self._loss_fn, list) else False
        self._train_losses = []
        self._val_losses = []
        self.lrs = set()

        self._train_data = None
        self._val_data = None
        self._train_step = self._train_step_fn()
        self._val_step = self._val_step_fn()

        self._total_epochs = 0

        self._lr_scheduler = None
        self._STEP_SCHEDULER = False
        self._BATCH_SCHEDULER = False

        self._filename = None
        self.names = None

    def _train_step_fn(self):
        """
        Defines the training step for a single batch.

        Returns:
            function: A function that computes the loss and accuracy for a batch.
        """
        def _step(x, y):
            self.model.train()

            x = x.to(self.device)
            y_pred = self.model(x)

            loss = 0
            if self.multi_task:
                losses = []
                for i, loss_fn in enumerate(self._loss_fn):
                    y[i] = y[i].to(self.device)
                    if isinstance(loss_fn, nn.BCELoss) or isinstance(loss_fn, nn.BCEWithLogitsLoss):
                        temp_loss = loss_fn(y_pred[i].squeeze(1), y[i])
                    else:
                        temp_loss = loss_fn(y_pred[i], y[i])
                    losses.append(temp_loss.item())
                    loss += temp_loss
            else:
                y = y.to(self.device)
                if isinstance(self._loss_fn, nn.BCELoss) or isinstance(self._loss_fn, nn.BCEWithLogitsLoss):
                    loss = self._loss_fn(y_pred, y.unsqueeze(1))
                else:
                    loss = self._loss_fn(y_pred, y)

            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            if self._BATCH_SCHEDULER:
                self._lr_scheduler.step()

            acc = self._accuracy(x, y)

            del x, y, y_pred
            if self.multi_task:
                return loss.item(), *losses, *acc
            else:
                return loss.item(), acc
        return _step

    def _val_step_fn(self):
        """
        Defines the validation step for a single batch.

        Returns:
            function: A function that computes the loss and accuracy for a batch.
        """
        def _step(x, y):
            self.model.eval()
            x = x.to(self.device)
            y_pred = self.model(x)

            loss = 0
            if self.multi_task:
                losses = []
                for i, loss_fn in enumerate(self._loss_fn):
                    y[i] = y[i].to(self.device)
                    if isinstance(loss_fn, nn.BCELoss) or isinstance(loss_fn, nn.BCEWithLogitsLoss):
                        temp_loss = loss_fn(y_pred[i].squeeze(1), y[i])
                    else:
                        temp_loss = loss_fn(y_pred[i], y[i])
                    losses.append(temp_loss.item())
                    loss += temp_loss
            else:
                y = y.to(self.device)
                if isinstance(self._loss_fn, nn.BCELoss) or isinstance(self._loss_fn, nn.BCEWithLogitsLoss):
                    loss = self._loss_fn(y_pred, y)
                else:
                    loss = self._loss_fn(y_pred, y)
            acc = self._accuracy(x, y)

            del y, x, y_pred

            if self.multi_task:
                return loss.item(), *losses, *acc
            else:
                return loss.item(), acc
        return _step

    def _mini_batch(self, validation=False):
        """
        Processes a mini-batch of data (training or validation).

        Args:
            validation (bool, optional): Whether to use the validation dataset. Defaults to False.

        Returns:
            tuple: Average loss and task-specific metrics (for multi-task learning).
        """
        if validation:
            dataloader = self._val_data
            step_fn = self._val_step
        else:
            dataloader = self._train_data
            step_fn = self._train_step

        if self.multi_task:
            try:
                infos = [0] * (len(self.names) * 2)
            except:
                raise ValueError('Set up task names: set_tasks_names(name1, name2, ...)')

        loss, total_acc = 0, 0
        for images, y in tqdm(dataloader):
            if self.multi_task:
                loss_batch, *inf = step_fn(images, y)
            else:
                loss_batch, acc = step_fn(images, y)
            loss += loss_batch

            if self.multi_task:
                for i in range(len(self.names) * 2):
                    infos[i] += inf[i]
            else:
                total_acc += acc
        del images, y
        if self.multi_task:
            return loss / len(dataloader), (np.array(infos) / len(dataloader)).tolist()

        return loss / len(dataloader), total_acc / len(dataloader)

    def set_lr_scheduler(self, scheduler):
        """
        Sets the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        """
        if scheduler.optimizer != self._optimizer:
            raise ValueError('Optimizer is not used in lr_scheduler')
        self._lr_scheduler = scheduler
        if isinstance(scheduler, StepLR) or \
                isinstance(scheduler, MultiStepLR) or \
                isinstance(scheduler, ReduceLROnPlateau):
            self._STEP_SCHEDULER = True
        elif isinstance(scheduler, CyclicLR) or isinstance(scheduler, OneCycleLR):
            self._BATCH_SCHEDULER = True

    def train(self, epochs, seed=42, display_table=True):
        """
        Trains the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            display_table (bool, optional): Whether to display metrics after each epoch. Defaults to True.
        """
        self._set_seed(seed)
        last_loss = None

        # Initialize a DataFrame to store training metrics
        columns = ['Epoch', "Training Loss", "Validation Loss"]
        if self.multi_task:
            for task in self.names:
                name_train = task + " Loss (Train)"
                name_val = task + " Loss (Val)"
                columns.append(name_train)
                columns.append(name_val)

            for task in self.names:
                name_train = task + " Acc (Train)"
                name_val = task + " Acc (Val)"
                columns.append(name_train)
                columns.append(name_val)

        else:
            columns.append("Train Acc")
            columns.append("Val Acc")
        
        if self._STEP_SCHEDULER:
            columns.append('Learning Rate')

        metrics_df = pd.DataFrame(columns=columns)

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            try:
                self._total_epochs += 1

                # Perform training step
                loss, train_inf = self._mini_batch()
                self._train_losses.append(loss)

                # Perform validation step
                with torch.no_grad():
                    val_loss, val_inf = self._mini_batch(validation=True)
                    self._val_losses.append(val_loss)

                # Step the learning rate scheduler if applicable
                if self._STEP_SCHEDULER:
                    if isinstance(self._lr_scheduler, ReduceLROnPlateau):
                        self._lr_scheduler.step(val_loss)
                    else:
                        self._lr_scheduler.step()
                    if self._lr_scheduler.optimizer.param_groups[0]['lr'] not in self.lrs:
                        self.lrs.add(self._lr_scheduler.optimizer.param_groups[0]['lr'])
                        print('Learning rate changed to --->', self._lr_scheduler.optimizer.param_groups[0]['lr'])

                # Save the best model checkpoint
                if last_loss is None or last_loss > val_loss:
                    last_loss = val_loss
                    if self._filename:
                        self.save_checkpoint(f'best_{self._filename}')
                    else:
                        self.save_checkpoint('best')

                if self.multi_task:
                    # Update the metrics DataFrame
                    new_row = {
                        "Epoch": self._total_epochs,
                        "Training Loss": round(loss, 4),
                        "Validation Loss": round(val_loss, 4),
                    }
                    idx = 2
                    for i in range(len(self.names)):
                        idx += 1
                        print(train_inf[i], idx)
                        print(columns[idx + len(self.names) * 2])
                        new_row[columns[idx]] = f"{train_inf[i]:.4f}"
                        new_row[columns[idx + len(self.names) * 2]] = f"{train_inf[i + len(train_inf) // 2]:.2f}%"
                        idx += 1
                        new_row[columns[idx]] = f"{val_inf[i]:.4f}"
                        new_row[columns[idx + 2 * len(self.names)]] = f"{val_inf[i + len(train_inf) // 2]:.2f}%"

                else:
                    new_row = {
                        "Epoch": self._total_epochs,
                        "Training Loss": round(loss, 4),
                        "Validation Loss": round(val_loss, 4),
                        "Train Acc": f"{train_inf:.2f}%",
                        "Val Acc": f"{val_inf:.2f}%",
                    }

                if self._STEP_SCHEDULER:
                    new_row['Learning Rate'] = self._lr_scheduler.optimizer.param_groups[0]['lr']
                    
                metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)

                # Optionally display table with results after each epoch
                if display_table:
                    clear_output(wait=True)
                    display(HTML(metrics_df.to_html()))

            except KeyboardInterrupt as e:
                if self._filename:
                    self.save_checkpoint(f'last_{self._filename}')
                else:
                    self.save_checkpoint('last')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ValueError('Training Interrupted by the User')
            except Exception as e:
                print('Fehler beim Training ', e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e

            # Save the most recent checkpoint
            if self._filename:
                self.save_checkpoint(f'last_{self._filename}')
            else:
                self.save_checkpoint('last')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_dataloaders(self, train_data, val_data=None):
        """
        Sets the training and validation DataLoaders.

        Args:
            train_data (torch.utils.data.DataLoader): DataLoader for the training dataset.
            val_data (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Defaults to None.
        """
        self._train_data = train_data
        if val_data is not None:
            self._val_data = val_data

    def to(self, device):
        """
        Moves the model and optimizer to the specified device.

        Args:
            device (str): The device ('cuda' or 'cpu').
        """
        self.device = device

    def set_tasks_names(self, *args):
        """
        Sets the names of tasks (for multi-task learning).

        Args:
            *args: Names of the tasks.
        """
        self.names = args

    @torch.no_grad()
    def _accuracy(self, x, y):
        """
        Computes accuracy for a batch of data.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            float or list: Accuracy for each task (for multi-task learning).
        """
        def binary_class(pred, target):
            pred = (pred > 0.5).squeeze()
            is_correct = (pred == target).sum().item()
            acc = (is_correct / len(target)) * 100
            return acc

        def multi_class(pred, target):
            pred = torch.argmax(pred, dim=1)
            is_correct = (pred == target).sum().item()
            acc = is_correct / target.size(0) * 100
            return acc

        self.model.eval()
        y_pred = self.model(x.to(self.device))

        if self.multi_task:
            acc = []
            for i in range(len(y_pred)):
                y[i] = y[i].to(self.device)
                if isinstance(self._loss_fn[i], nn.BCELoss) or isinstance(self._loss_fn[i], nn.BCEWithLogitsLoss):
                    acc.append(multi_class(y_pred[i], y[i]))
                else:
                    acc.append(multi_class(y_pred[i], y[i]))
        else:
            y = y.to(self.device)
            if isinstance(self._loss_fn, nn.BCELoss) or isinstance(self._loss_fn, nn.BCEWithLogitsLoss):
                acc = binary_class(y_pred, y)
            else:
                acc = multi_class(y_pred, y)
        del y, x, y_pred
        return acc

    def _set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed.
        """
        if self.device != 'cpu':
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    def save_checkpoint(self, filename):
        """
        Saves the model checkpoint.

        Args:
            filename (str): The filename for the checkpoint.
        """
        checkpoint = {'epoch': self._total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self._optimizer.state_dict(),
                      'loss': self._train_losses,
                      'val_loss': self._val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """
        Loads the model checkpoint.

        Args:
            filename (str): The filename of the checkpoint.
        """
        checkpoint = torch.load(filename, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._total_epochs = checkpoint['epoch']
        self._train_losses = checkpoint['loss']
        self._val_losses = checkpoint['val_loss']
        self.model.train()

    def set_filename(self, filename):
        """
        Sets the filename prefix for saving checkpoints.

        Args:
            filename (str): The filename prefix.
        """
        self._filename = filename

    def plot_losses(self):
        """
        Plots the training and validation losses.

        Returns:
            matplotlib.figure.Figure: A Matplotlib figure.
        """
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self._train_losses, label='Training Loss', c='b')
        plt.plot(self._val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def conf_mat_class_report(self, classes, validation=False, report=False):
        """
        Computes and displays the confusion matrix and classification report.

        Args:
            classes (list): List of class labels.
            validation (bool, optional): Whether to use the validation dataset. Defaults to False.
            report (bool, optional): Whether to display the classification report. Defaults to False.
        """
        torch.manual_seed(42)

        # Load the appropriate checkpoint
        checkpoint_name = 'best' if self._filename is None else f'best_{self._filename}'
        try:
            self.load_checkpoint(checkpoint_name)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print('The current model with his parameters will be used')

        # Prepare data containers
        if self.multi_task:
            y_pred = [[] for _ in range(len(self.names))]
            y_true = [[] for _ in range(len(self.names))]
        else:
            y_pred = []
            y_true = []

        # Select the appropriate dataloader
        dataloader = self._val_data if validation else self._train_data

        # Disable gradient calculations for inference
        with torch.inference_mode():
            self.model.eval()
            for x, y in tqdm(dataloader):
                x = x.to(self.device)

                # Forward pass
                outputs = self.model(x)

                if self.multi_task:
                    # Handle multi-task predictions
                    for i, pred in enumerate(outputs):
                        if pred.size(1) > 2:  # Multi-class classification
                            pred_labels = pred.argmax(dim=1).cpu().numpy()
                        else:  # Binary classification
                            pred_labels = (pred > 0.5).float().cpu().numpy()

                        y_pred[i].extend(pred_labels)
                        y_true[i].extend(y[i].cpu().numpy())
                else:
                    # Handle single-task predictions
                    if outputs.size(1) > 2:  # Multi-class classification
                        pred_labels = outputs.argmax(dim=1).cpu().numpy()
                    else:  # Binary classification
                        pred_labels = (outputs > 0.5).float().cpu().numpy()

                    y_pred.extend(pred_labels)
                    y_true.extend(y.cpu().numpy())

                del x, y
        # Restore the model to training mode
        self.model.train()

        # Visualization and reports
        if self.multi_task:
            for i, (true_labels, pred_labels) in enumerate(zip(y_true, y_pred)):
                self._plot_confusion_matrix(true_labels, pred_labels, classes[i], report)
        else:
            self._plot_confusion_matrix(y_true, y_pred, classes, report)

    def _plot_confusion_matrix(self, y_true, y_pred, class_labels, report):
        """
        Helper function to plot the confusion matrix and display the classification report.

        Args:
            y_true (list or np.ndarray): True labels.
            y_pred (list or np.ndarray): Predicted labels.
            class_labels (list): List of class labels.
            report (bool): Whether to display the classification report.
        """
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(
            cf_matrix,
            index=[i for i in class_labels],
            columns=[i for i in class_labels]
        )
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        if report:
            print(classification_report(y_true, y_pred, target_names=class_labels))