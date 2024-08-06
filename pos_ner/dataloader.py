import pandas as pd
import torch
from torch import Tensor
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Data:
    vocab_size: int = 0
    pos_size: int = 0
    ner_size: int = 0
    pos_encoder = LabelEncoder()
    ner_encoder = LabelEncoder()
    vocabs = 0

    def __init__(self, csv_path_or_df:str, truncate:int=None, test_size:float=0.1)->None:
        # Load either CSV file or DataFrame
        if isinstance(csv_path_or_df, pd.DataFrame):
            dataframe = csv_path_or_df
        else:
            dataframe = pd.read_csv(csv_path_or_df).dropna()

        # Split dataframe into train and test
        if truncate:
            train_df, test_df = train_test_split(dataframe, test_size=test_size)
            train_df = train_df[:truncate * test_size]
            test_df = test_df[:truncate * test_size]
        else:
            train_df, test_df = train_test_split(dataframe, test_size=test_size)

        # Get vocabs from the entire dataframe
        Data.vocabs = set(["<UNK>"] + dataframe["Word"].tolist())
        self.train_df.head()
        # Set data attributes
        Data.vocab_size = len(self.vocabs) + 1
        Data.pos_size = len(self.train_df.iloc[:, 1].unique())
        Data.ner_size = len(self.train_df.iloc[:, 2].unique())

    @classmethod
    def index2word(cls, indices:Optional[int])->str:
        i2w = {i+1: w for i, w in enumerate(cls.vocabs)}
        return " ".join([i2w[i] for i in indices])

    @classmethod
    def word2index(cls, words:str)->List[int]:
        w2i = {w:i+1 for i, w in enumerate(cls.vocabs)}
        if isinstance(words, str):
            # If the input is a string, split it into words
            return [w2i.get(word, w2i["<UNK>"]) for word in words.split()]
        # If the input is a list of words
        return [w2i.get(word, w2i["<UNK>"]) for word in words]

    def build_sequence(self, df:pd.DataFrame, seq_len:int, col:str)->Tensor:
        """
        This block of code is responsible for building
        sequences from a dataframe based on a specified column.

        Args:
            df: The dataframe from where the sequences are generated.
            seq_len: The length of each sequence.
            col: The name of the column from which the sequences are generated.
        Returns:
            torch.Tensor: A tensor of sequences with shape (num_sequences, seq_len).
        """
        if col in [df.columns[1], df.columns[2]]:
            if col == df.columns[1]:
                df[col] = Data.pos_encoder.fit_transform(df[df.columns[1]])
            else:
                df[col] = Data.ner_encoder.fit_transform(df[df.columns[2]])

        sequences = []
        if isinstance(df[col].iloc[0], str):
            col_data = Data.word2index(df[col].tolist())
        else:
            col_data = df[col].tolist()

        for i in range(len(df)-seq_len):
            sequence = (col_data[i:i+seq_len])
            if type(col_data[i]) == str:
                sequences.append(sequence)
                print(sequence)
            else:
                sequences.append(sequence)
        return torch.tensor(sequences)


    def plot_labels(self, row:str)->None:
        self.train_df[row].value_counts().plot(kind="bar")
        plt.title(f"Label Distribution for {row}")
        plt.xlabel(row)
        plt.ylabel("Count")
        plt.show()

    def build_dataloader(self, batch_size:int, seq_length:int, train_size:float=0.8
        )->Tuple[DataLoader, DataLoader]:
        """
        This block of code is responsible for building train and val dataloaders.

        Args:
            batch_size (int): The batch size.
            seq_length (int): The length of each sequence.
            train_size (float): The proportion of train data.
        Returns:
            A tuple of train and val dataloaders.
        """
        input_seq = self.build_sequence(
            self.train_df, seq_length, self.train_df.columns[0])
        target_pos = self.build_sequence(
            self.train_df, seq_length, self.train_df.columns[1])
        target_ner = self.build_sequence(
            self.train_df, seq_length, self.train_df.columns[2])
        targets = torch.stack((target_pos, target_ner), dim=1)
        dataset = TensorDataset(input_seq, targets)
        # Split train and val
        train_size = int(train_size * len(dataset))
        val_size = int((len(dataset) - train_size))
        train, val = random_split(dataset, [train_size, val_size])
        # Load train and val to pytorch dataloader
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=batch_size)
        return train_loader, val_loader



if __name__ == "__main__":
    data = Data("Data/train_test_df.csv", truncate=1000)
    tokens = Data.word2index("পটুয়াখালী আগস্ট রাতে")
    print(tokens)
    print(data.index2word(tokens))
    print("Training Data Shape: ", data.train_df.shape)
    print("Testing Data Shape: ", data.test_df.shape)
    data.plot_labels("POS")
    print(Data.vocab_size)