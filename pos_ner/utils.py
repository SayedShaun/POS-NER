from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
    )
import torch
import seaborn as sns
from pos_ner.dataloader import Data
import warnings
warnings.filterwarnings("ignore")


def train_epoch(model, loss_fn, optimizer, train_data, device)->float:
    """
    Args:
        model: The model to be used for training.
        loss_fn: The loss function to be used for training.
        optimizer: The optimizer to be used for training.
        train_data: The training data.
        device: The device to be used for training.
    Returns:
        avg_train_loss: The average training loss.
    """
    avg_train_loss = 0.0
    model.train()
    for inputs, targets in train_data:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        # Fetch two targets
        pos_y, ner_y = targets[:,0,:].to(device), targets[:,1,:].to(device)
        pos_logits, ner_logits = model(inputs)
        pos_loss = loss_fn(pos_logits.flatten(0, 1), pos_y.reshape(-1))
        ner_loss = loss_fn(ner_logits.flatten(0, 1), ner_y.reshape(-1))
        # Combine losses
        loss = pos_loss + ner_loss
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.item()
    return avg_train_loss / len(train_data)

def validate_epoch(model, loss_fn, val_data, device)->float:
    """
    Args:
        model: The model to be used for validation.
        loss_fn: The loss function to be used for validation.
        val_data: The validation data.
        device: The device to be used for validation.
    Returns:
        avg_val_loss: The average validation loss.
    """
    avg_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for val_inputs, val_targets in val_data:
            val_inputs = val_inputs.to(device)
            # Fetch two targets
            val_pos_y, val_ner_y = val_targets[:,0,:].to(device), val_targets[:,1,:].to(device)
            val_pos_logits, val_ner_logits = model(val_inputs)
            val_pos_loss = loss_fn(val_pos_logits.flatten(0, 1), val_pos_y.reshape(-1))
            val_ner_loss = loss_fn(val_ner_logits.flatten(0, 1), val_ner_y.reshape(-1))
            # Combine losses
            val_loss = val_pos_loss + val_ner_loss
            avg_val_loss += val_loss.item()
    return avg_val_loss / len(val_data)


def classification_reports(model, test_df:pd.DataFrame, cm:bool=False)->None:
    """
    Args:
        test_df: Test dataframe
        cm: A boolean variable to decide whether to plot confusion matrix
    Returns:
        None
    """
    input_text = test_df["Word"].tolist()
    pos_pred, ner_pred = model.predict(input_text)
    test_df["POS"] = Data.pos_encoder.transform(test_df["POS"])
    test_df["NER"] = Data.ner_encoder.transform(test_df["NER"])
    pos_true = test_df["POS"].tolist()
    ner_true = test_df["NER"].tolist()

    metrics = {
        'Accuracy': [
            accuracy_score(pos_true, pos_pred),
            accuracy_score(ner_true, ner_pred)
            ],
        'Precision': [
            precision_score(pos_true, pos_pred, average='macro'),
            precision_score(ner_true, ner_pred, average='macro')
            ],
        'Recall': [
            recall_score(pos_true, pos_pred, average='macro'),
            recall_score(ner_true, ner_pred, average='macro')
            ],
        'F1 Score': [
            f1_score(pos_true, pos_pred, average='macro'),
            f1_score(ner_true, ner_pred, average='macro')
            ]
        }
    # Create a dataframe from the metrics dictionary
    metrics_df = pd.DataFrame(metrics, index=['POS', 'NER'])
    plt.figure(figsize=(7, 3))
    sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.title('Performance Metrics')
    plt.show()

    if cm:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        pos_cm = confusion_matrix(pos_true, pos_pred)
        sns.heatmap(pos_cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
        ax[0].set_title('POS Confusion Matrix')
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('True')

        ner_cm = confusion_matrix(ner_true, ner_pred)
        sns.heatmap(ner_cm, annot=True, fmt='d', cmap='Blues', ax=ax[1])
        ax[1].set_title('NER Confusion Matrix')
        ax[1].set_xlabel('Predicted')
        ax[1].set_ylabel('True')
        plt.tight_layout()
        plt.show()

