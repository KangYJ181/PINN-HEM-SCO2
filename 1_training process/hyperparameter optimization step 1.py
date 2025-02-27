import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import matplotlib.pyplot as plt
import time


# Pseudo random, guaranteed repeatability
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define the purely data-driven network
class DataDriven(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DataDriven, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)


        # Ensure that the output contains the interval [1.5, 7.3] after the inverse normalization
        # The minimum pressure in training concentration is 7.5 MPa and the maximum wis 14.01 MPa
        out = (1.8691 - 0.06231) / 2 * torch.sin(out) - (1.8691 + 0.06231) / 2
        return out

    def initialize_weights(self):
        # Initialize the weight and bias of the fully connected layer
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)


def custom_loss(pc_true, p_c):
    loss = nn.MSELoss()(p_c, pc_true)
    return loss

start_time = time.time()
def train_and_evaluate_DataDriven(learning_rate, weight_decay, hidden_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model = DataDriven(input_size, hidden_size, output_size).to(device)
    model.initialize_weights()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            optimizer.zero_grad()
            p_c = model(batch_x)
            loss = custom_loss(batch_y, p_c)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        # Save the weight and bias of the purely data-driven model
        torch.save(model.state_dict(), 'data_driven_model_weights.pth')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{execution_time}seconds")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='blue')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Prediction results on validation set
    model.eval()
    with torch.no_grad():
        p_c_preds2 = []
        p_c_trues2 = []
        relative_errors = []
        for i, (batch_x, batch_y) in enumerate(valid_loader):
            batch_x = batch_x.to(device)
            p_c_pred2 = model(batch_x)
            p_c_true2 = batch_y
            p_c_pred2 = p_c_pred2 * (max_val_pre - min_val_pre) + min_val_pre
            p_c_true2 = p_c_true2 * (max_val_pre - min_val_pre) + min_val_pre
            relative_error = (p_c_pred2 - p_c_true2) / p_c_true2 * 100
            p_c_preds2.append(p_c_pred2.cpu().numpy())
            p_c_trues2.append(p_c_true2.cpu().numpy())
            relative_errors.append(relative_error.cpu().numpy())

    p_c_preds2 = np.concatenate(p_c_preds2).flatten()
    p_c_trues2 = np.concatenate(p_c_trues2).flatten()
    relative_errors = np.concatenate(relative_errors).flatten()
    print(np.mean(np.abs(relative_errors)))
    result_df = pd.DataFrame({'True_value': p_c_trues2, 'Predicted_value': p_c_preds2})
    result_df.to_csv('validation_results.csv', index=False)


# Adjustable parameters
########################################################################################################################
input_size = 4
output_size = 1
epochs = 5000
batch_size = 32
num_train = 160  # The number of samples of the training set in the data set
learning_rate = 0.0001
weight_decay = 1e-5
hidden_size = 22
########################################################################################################################


seed = 42
set_seed(seed)
device = torch.device("cpu")

# Set the REFPROP
RP = REFPROPFunctionLibrary('C:/Program Files (x86)/REFPROP')
iMass = 1  # 1 represents the mass basis
iFlag = 0  # 0 represents the standard calculation process
MASS_BASE_SI = RP.GETENUMdll(iFlag, "MASS BASE SI").iEnum  # Only REFPROP 10.0 can use this function

data = pd.read_csv('training&validation_DataDriven.csv')  # Load the data
x_raw = data.drop('p_c', axis=1).values
pc_raw = data['p_c'].values.reshape(-1, 1)

# Split the data into training and validation sets
x_train_raw = x_raw[:num_train]
pc_train_raw = pc_raw[:num_train]
x_val_raw = x_raw[num_train:]
pc_val_raw = pc_raw[num_train:]

# Data preprocessing: scale the training data to the 0~1 range
scaler_x = MinMaxScaler()
x_train_scaled = scaler_x.fit_transform(x_train_raw)
x_val_scaled = scaler_x.transform(x_val_raw)

# The maximum and minimum values of the pressure in the training set
min_val_pre = scaler_x.data_min_[1]
max_val_pre = scaler_x.data_max_[1]

pc_train_scaled = (pc_train_raw - min_val_pre) / (max_val_pre - min_val_pre)
pc_val_scaled = (pc_val_raw - min_val_pre) / (max_val_pre - min_val_pre)

# Convert to torch tensors
x_train = torch.from_numpy(x_train_scaled).float()
pc_train = torch.from_numpy(pc_train_scaled).float()
x_val = torch.from_numpy(x_val_scaled).float()
pc_val = torch.from_numpy(pc_val_scaled).float()

# Create a DataLoader object to pair input with output
train_dataset = torch.utils.data.TensorDataset(x_train, pc_train)
val_dataset = torch.utils.data.TensorDataset(x_val, pc_val)


# Training

train_and_evaluate_DataDriven(learning_rate, weight_decay, hidden_size)