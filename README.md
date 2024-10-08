# Part-Of-Speech and Name Entity Recognition System

This project is a part-of-speech and named entity recognition system. It is built using PyTorch and trained on a dataset of Bengali text. The system can identify the part-of-speech (POS) and named entity (NER) of words in the text.

The system consists of a data loader, a model architecture, and a set of utility functions. The data loader loads the data from a CSV file into a dataframe and prepares it for training. The model architecture is a multi-task model that predicts POS and NER simultaneously. The utility functions are used to calculate metrics such as accuracy, precision, recall, and F1-score.

The system can be trained and used to analyze various languages and identify the part-of-speech and named entity of words in the text.

## Usage
Install the package first
```bash
pip install git+https://github.com/SayedShaun/POS-NER.git
``` 
Here is the step by step guide to use the package
``` python
# Import the necessary libraries
import torch
from pos_ner import GruMultiTaskModel, Config, Data

# First load the data e.g. csv or dataframe
data = Data(dataframe or "csv file path")
# Configure the hyperparameter as necessary
config = Config(vocab_size=Data.vocab_size, pos_size=Data.pos_size, ner_size=Data.ner_size, n_ctx=100)
# Build the dataloader (Batch Size, Seq Length, Train Size)
train_ds, val_ds = data.build_dataloader(256, config.n_ctx, 0.8)
# Select the model
model = GruMultiTaskModel(config).to(device)
# Define the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Train the model with necessary parameters
model.fit(10, loss_fn, optimizer, train_ds, val_ds, callbacks=True)
# Get predictions
model.predict("শনিবার ২৭ আগস্ট রাতে পটুয়াখালী সদর থানার ভারপ্রাপ্ত কর্মকর্তা")
# For In-depth Reports
model.test_report(data.test_df.head(100))
```
## Model Details:
We have tested two models architectures Gru Based Model and Transformer Based Model. Both Model architectures are trained on the same dataset for 10 epochs with callbacks. In this particular task Gru based model gives better performance. Gru based model gives slightly better accuracy than transformer based model.

## Results
We have tested Gru based model and Transformer based model with same dataset for 10 epochs with callbacks. Gru gives slightly better performance. Though Transformer model known for better performance with very large dataset.Here is the performance of Gru based model and Transformer based model's performance. for training details please checkout the Results directory.

!["alt text"](Results/gru_performance.png) 
![alt text](Results/transformer_performance.png)

## Recommendations:
While training both models with minimal hyperparameter tuning, consider exploring opportunities for further tuning to potentially enhance performance. Additionally, incorporating a larger dataset can contribute positively to the model's performance. Also a warning to use GPU for training on large datasets.


## Web Interface
A streamlit web app is available for you to train complex model without witting single lines of code. To use web interface to train model please clone the repository and run `streamlit run app.py`

```bash
https://github.com/SayedShaun/POS-NER.git
```
![image](https://github.com/user-attachments/assets/bef99035-86dc-4c1a-857f-33e4116f2e0f)


## Docker Use ( Streamlit App)
To build docker image with from this repository run, make sure you have docker installed on your system. Then run the following command
```bash
docker build -t <image_name> .
```

To publish the docker image to docker hub run, make sure you have docker installed on your system. Then run the following command
```bash
docker tag <image_id> <username>/<image_name>
docker push <username>/<image_name>
```

To pull the docker image from docker hub run, make sure you have docker installed on your system. Then run the following command
```bash
docker pull <username>/<image_name>
```

To run the docker image run, make sure you have docker installed on your system. Then run the following command
```bash
docker run -p 8501:8501 <username>/<image_name>
```

## Contributors
If you have any suggestion or problem, please feel free to open an issue on [Github](https://github.com/SayedShaun/POS-NER/issues). Thank you.





