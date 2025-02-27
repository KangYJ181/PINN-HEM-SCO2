import shap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))
        self.e = nn.Parameter(torch.randn(1))
        self.f = nn.Parameter(torch.randn(1))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, nonlinearity='tanh', batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size - 6)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size)
        x = x.unsqueeze(1)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        coefficients = torch.stack([self.a, self.b, self.c, self.d, self.e, self.f]).view(1, -1)
        coefficients = coefficients.repeat(out.size(0), 1)
        out = torch.cat((out, coefficients), dim=1)
        out = self.fc2(out)
        out = out * (max_val_mas - min_val_mas) + min_val_mas
        return out


def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        w = model(x_tensor)

    return w.numpy()

data_train = pd.read_csv('training&validation_PINN.csv')
data_test = pd.read_csv('test data.csv')
data_BGD = pd.read_csv('background data.csv')

x_raw = data_train.drop('mas', axis=1).values
G_exp_raw = data_train['mas'].values.reshape(-1, 1)


x_test_raw = data_test.drop('mas', axis=1).values
x_BGD_raw = data_BGD.drop('mas', axis=1).values

scaler_x = MinMaxScaler()
x_train_scaled = scaler_x.fit_transform(x_raw)
x_test_scaled = scaler_x.transform(x_test_raw)
x_BGD_scaled = scaler_x.transform(x_BGD_raw)

scaler_G = MinMaxScaler()
G_exp_train_scaled = scaler_G.fit_transform(G_exp_raw)

x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test = torch.tensor(x_test_scaled, dtype=torch.float32)
G_exp_train = torch.tensor(G_exp_train_scaled, dtype=torch.float32)

min_val_mas = scaler_G.data_min_
max_val_mas = scaler_G.data_max_

model = RNN(input_size = 5, hidden_size = 9, output_size = 1)
model.load_state_dict(torch.load('PINN-EF.pth'))

explainer = shap.KernelExplainer(model_predict, x_BGD_scaled)
shap_values = explainer.shap_values(x_test_scaled[0:10],nsamples = 100)
shap_values_array = shap_values.reshape(10, 5)
shap_df = pd.DataFrame(shap_values_array, columns=[f"Feature_{i+1}" for i in range(shap_values_array.shape[1])])
shap_df.to_csv('output.csv', index=False)

