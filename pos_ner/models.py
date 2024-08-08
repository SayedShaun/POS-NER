from dataclasses import dataclass
import torch
import pandas as pd
from tqdm import tqdm
from pos_ner.utils import (
    classification_reports, 
    train_epoch, 
    validate_epoch
    )
from pos_ner.dataloader import Data
from torch import nn, Tensor
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from typing import List, Tuple, Optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Config:
    vocab_size: int = 0
    pos_size: int = 0
    ner_size: int = 0
    gru_hidden: int = 128
    gru_layers: int = 1
    bidirectional: bool = False
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    n_ctx: int = 20
    d_ff: int = 128
    head_dim: int = d_model//n_heads
    dropout_p: float = 0.5
    

class GruMultiTaskModel(nn.Module):
    def __init__(self, config: Config)->None:
        super(GruMultiTaskModel, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.gru_hidden)
        self.gru = nn.GRU(config.gru_hidden, config.gru_hidden,
            batch_first=True, num_layers=config.gru_layers,
            bidirectional=config.bidirectional
            )
        if config.bidirectional:
            self.pos_output = nn.Linear(config.gru_hidden*2, config.pos_size)
            self.ner_output = nn.Linear(config.gru_hidden*2, config.ner_size)
        else:
            self.pos_output = nn.Linear(config.gru_hidden, config.pos_size)
            self.ner_output = nn.Linear(config.gru_hidden, config.ner_size)
        self.dropout = nn.Dropout(config.dropout_p)
        self.name = "GruMultiTaskModel"

    def forward(self, X: Tensor)->Tuple[Tensor, Tensor]:
        """
        Args:
            X: A tensor of shape (batch_size, seq_length)
        Returns:
            pos_output: A tensor of shape (batch_size, seq_length, pos_size)
            ner_output: A tensor of shape (batch_size, seq_length, ner_size)
        """
        embedded = self.dropout(self.embedding(X))
        # Unpack GRU Tuple output
        out, hidden = self.gru(embedded)
        # print("GRU OUT: ", gru_out.shape, "GRU HIDDEN: ", gru_hidden.shape)
        pos_output = self.pos_output(self.dropout(out))
        ner_output = self.ner_output(self.dropout(out))
        return pos_output, ner_output

    def trainable_parameters(self):
        params = 0
        for param in self.parameters():
            if param.requires_grad:
                params += param.numel()
        return f"Trainable Parameters: {params/1e6} M"

    def fit(self,
            epochs,
            loss_fn,
            optimizer,
            train_data,
            val_data=None,
            verbose=False,
            plot=False,
            patience: int = 5,
            callbacks: bool= False)->None:
        """
        Args:
            epochs (int): The number of epochs to train the model
            loss_fn: Pytorch loss function e.g. CrossEntropyLoss
            optimizer: Pytorch optimizer e.g. Adam, RMSprop, etc
            train_data: Training data, a tuple of (X, (y1, y2))
            val_data: Validation data, a tuple of (X, (y1, y2))
            verbose: A boolean variable to decide whether to print loss history
            plot: A boolean variable to decide whether to plot loss history
        Returns:
            None
        """
        # Define training and validation loops
        train_loss_arr, val_loss_arr = [], []
        patiences = 0
        best_val_loss = float('inf')
        for epoch in tqdm(range(epochs)):
            train_loss = train_epoch(self, loss_fn, optimizer, train_data, device)
            val_loss = validate_epoch(self, loss_fn, val_data, device)
            if callbacks:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patiences = 0  # Reset patience since we have a new best validation loss
                else:
                    patiences += 1  # Increment patience if validation loss did not improve

            if patiences > patience:
                print("Early stopping triggered")
                break

            # Show training and validation loss history
            if verbose:
                if verbose and val_data is not None:
                    print(f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
                else:
                    print(f"Train Loss: {train_loss:.3f}")
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)

        if plot:
            # A boolean variable to decide whether to plot loss history
            plt.figure(figsize=(7, 3))
            plt.plot(train_loss_arr, label='Train Loss')
            if val_data is not None:
                plt.plot(val_loss_arr, label='Val Loss')
            plt.title(f'Loss History | {self.name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def predict(self, input_text:Optional[List[str]])->Tuple[List[int], List[int]]:
        """
        Args:
            input_text: A list of strings or a single string
        Returns:
            A tuple of predicted POS tags and predicted NER tags
        """
        tokens = Data.word2index(input_text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            pos_output, ner_output = self(tokens)
            pos_probs = torch.softmax(pos_output, dim=-1)
            ner_probs = torch.softmax(ner_output, dim=-1)
            pos_pred = torch.argmax(pos_probs, dim=-1)
            ner_pred = torch.argmax(ner_probs, dim=-1)
        return pos_pred.cpu().numpy().squeeze(), ner_pred.cpu().numpy().squeeze()

    def test_report(self, test_df:pd.DataFrame, cm:bool=False)->None:
        """
        Args:
            test_df: Test dataframe
            cm: A boolean variable to decide whether to plot confusion matrix
        Returns:
            None
        """
        classification_reports(self, test_df, cm)


class TransformerMultitaskModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(TransformerMultitaskModel, self).__init__()
        self.word_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.n_ctx, config.d_model)
        block = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.d_ff,
            batch_first=True, dropout=config.dropout_p
            )
        self.encoder = nn.TransformerEncoder(block, config.n_layers)
        self.pos_layer = nn.Linear(config.d_model, config.pos_size)
        self.ner_layer = nn.Linear(config.d_model, config.ner_size)
        self.name = "TransformerMultitaskModel"

    def forward(self, X:Tensor)->Tuple[Tensor, Tensor]:
        w_embed = self.word_embed(X)
        position = torch.arange(0, X.shape[1]).unsqueeze(0).to(device)
        p_embed = self.position_embed(position)
        embeddings = w_embed + p_embed
        encoder = self.encoder(embeddings)
        pos_output = self.pos_layer(encoder)
        ner_output = self.ner_layer(encoder)
        return pos_output, ner_output

    def fit(self,
            epochs,
            loss_fn,
            optimizer,
            train_data,
            val_data=None,
            verbose=False,
            plot=False,
            patience: int = 5,
            callbacks: bool= False
            )->None:
        """
        Args:
            epochs (int): The number of epochs to train the model
            loss_fn: Pytorch loss function e.g. CrossEntropyLoss
            optimizer: Pytorch optimizer e.g. Adam, RMSprop, etc
            train_data: Training data, a tuple of (X, (y1, y2))
            val_data: Validation data, a tuple of (X, (y1, y2))
            verbose: A boolean variable to decide whether to print loss history
            plot: A boolean variable to decide whether to plot loss history
        Returns:
            None
        """
        # Define training and validation loops
        train_loss_arr, val_loss_arr = [], []
        patiences = 0
        best_val_loss = float('inf')
        for epoch in tqdm(range(epochs)):
            train_loss = train_epoch(self, loss_fn, optimizer, train_data, device)
            val_loss = validate_epoch(self, loss_fn, val_data, device)
            if callbacks:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patiences = 0  # Reset patience since we have a new best validation loss
                else:
                    patiences += 1  # Increment patience if validation loss did not improve

            if patiences > patience:
                print("Early stopping triggered")
                break

            # Show training and validation loss history
            if verbose:
                if verbose and val_data is not None:
                    print(f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
                else:
                    print(f"Train Loss: {train_loss:.3f}")
            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)

        if plot:
            # A boolean variable to decide whether to plot loss history
            plt.figure(figsize=(7, 3))
            plt.plot(train_loss_arr, label='Train Loss')
            if val_data is not None:
                plt.plot(val_loss_arr, label='Val Loss')
            plt.title(f'Loss History | {self.name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def predict(self, input_text:Optional[List[str]])->Tuple[List[int], List[int]]:
        """
        Args:
            input_text: A list of strings or a single string
        Returns:
            A tuple of predicted POS tags and predicted NER tags
        """
        tokens = Data.word2index(input_text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            pos_output, ner_output = self(tokens)
            pos_probs = torch.softmax(pos_output, dim=-1)
            ner_probs = torch.softmax(ner_output, dim=-1)
            pos_pred = torch.argmax(pos_probs, dim=-1)
            ner_pred = torch.argmax(ner_probs, dim=-1)
        return pos_pred.cpu().numpy().squeeze(0), ner_pred.cpu().numpy().squeeze(0)

    def test_report(self, test_df:pd.DataFrame, cm:bool=False)->None:
        """
        Args:
            test_df: Test dataframe
            cm: A boolean variable to decide whether to plot confusion matrix
        Returns:
            None
        """
        classification_reports(self, test_df, cm)


if __name__ == "__main__":
    data = Data(df)
    train_ds, val_ds = data.build_dataloader(512, 100, 0.8)
    config = Config(vocab_size=data.vocab_size, pos_size=data.pos_size, ner_size=data.ner_size)
    model = GruMultiTaskModel(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.fit(10, loss_fn, optimizer, train_ds, val_ds, callbacks=True, plot=True)
    model.test_report(data.test_df)