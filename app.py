# This is the code to run train the model using streamlit interface
import torch
import streamlit as st
from pos_ner.dataloader import Data
device = "cuda" if torch.cuda.is_available() else "cpu"
from pos_ner.models import GruMultiTaskModel, Config, TransformerMultitaskModel
st.set_option('deprecation.showPyplotGlobalUse', False)


st.header("POS-NER Trainer")
st.sidebar.header("Model's Hyperparameters")
selected_model = st.sidebar.selectbox("", ["GRU", "Transformer"])

# Hyperparameters
if selected_model == "GRU":
    n_ctx = st.sidebar.slider("Sequence Length", 10, 1000, 10)
    gru_hidden = st.sidebar.slider("GRU Hidden", 32, 1024, 32)
    gru_layers = st.sidebar.slider("GRU Layers", 1, 4, 1)
    bidirectional = st.sidebar.radio("Bidirectional", [False, True])
    dropout_p = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)
elif selected_model == "Transformer":
    d_model = st.sidebar.slider("Dimension of Model", 64, 512, 64)
    n_heads = st.sidebar.slider("Head Size", 1, 8, 4)
    n_layers = st.sidebar.slider("Number of Layers", 1, 8, 4)
    ff_d = st.sidebar.slider("Feed Forward Dim", 100, 1000, 100)
    dropout_p = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)

# Accepted csv file format
# There will be three columns ["Word", "POS", "NER"]
st.text("Accepted csv file format:")
st.dataframe({
    "Word": ["The", "quick", "brown", "fox", "jumps"],
    "POS": ["DT", "JJ", "JJ", "NN", "NN"],
    "NER": ["O", "O", "O", "O", "O"]
    }, use_container_width=True)


# File uploader for CSV
upload_csv = st.file_uploader("Upload CSV", type=["csv"])
# Load data function
def load_data(csv_path: str, batch_size: int = 32, train_val_size: float = 0.8):
    data = Data(csv_path_or_df=csv_path)
    if selected_model == "GRU":
        config = Config(
            vocab_size=data.vocab_size,
            pos_size=data.pos_size,
            ner_size=data.ner_size,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            bidirectional=bidirectional,
            dropout_p=dropout_p,
            n_ctx=n_ctx,
        )
    elif selected_model == "Transformer":
        config = Config(
            vocab_size=data.vocab_size,
            pos_size=data.pos_size,
            ner_size=data.ner_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=ff_d,
            dropout_p=dropout_p
        )
    train_ds, val_ds = data.build_dataloader(batch_size, config.n_ctx, train_val_size)
    if selected_model == "GRU":
        model = GruMultiTaskModel(config).to(device)
    elif selected_model == "Transformer":
        model = TransformerMultitaskModel(config).to(device)
    return data, model, config, train_ds, val_ds

# Set training parameters
def set_params():
    st.sidebar.header("Training Parameters")
    # Batch size
    batch_size = st.sidebar.slider("Batch Size", 32, 1024, 32)
    # Number of epochs
    epochs = st.sidebar.slider("Epochs", 1, 100, 10)
    # Learning rate
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.01)
    # Optimizer selection
    optimizer_box = st.sidebar.selectbox("Optimizer", 
                                         ["Adam", 
                                          "AdamW", 
                                          "SGD", 
                                          "RMSprop", 
                                          "Adagrad", 
                                          "Adadelta", 
                                          "Adamax"]
                                          )
    
    return batch_size, epochs, learning_rate, optimizer_box

# Initialize model and config
data, model, config, train_ds, val_ds = None, None, None, None, None
set_parameters = st.button("Set Training Parameters", use_container_width=True)
# If a CSV is uploaded
if upload_csv:
    if set_parameters:
        batch_size, epochs, learning_rate, optimizer_box = set_params()
        data, model, config, train_ds, val_ds = load_data(csv_path=upload_csv, batch_size=batch_size)
    else:
        batch_size, epochs, learning_rate, optimizer_box = set_params()
        data, model, config, train_ds, val_ds = load_data(csv_path=upload_csv)


    if optimizer_box == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_box == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_box == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_box == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_box == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_box == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_box == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    train_model = st.button("Train Model", use_container_width=True)
    if train_model:
        with st.spinner('Wait, Model is training...'):
            st.pyplot(model.fit(epochs=epochs, 
                            loss_fn=loss_fn, 
                            optimizer=optimizer, 
                            train_data=train_ds, 
                            val_data=val_ds, 
                            plot=True,
                            callbacks=True,
                            )
                    )
            st.success("Training Completed!")
        st.pyplot(fig=model.test_report(data.test_df.head(100)))



    
