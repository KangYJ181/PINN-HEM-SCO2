import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import matplotlib.pyplot as plt


# Define the network
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = (1.8691 - 0.06231) / 2 * torch.sin(out) - (1.8691 + 0.06231) / 2
        return out

    def physics(self, ldr, p_0, dia, tem, p_c):
        dp_list = []
        w_list = []
        z = [1.0]

        for i in range(p_0.size(0)):
            s_0 = RP.REFPROPdll('CO2', 'PT', 'S', RP.MASS_BASE_SI, iMass, iFlag, p_0[i] * 1e6, tem[i] + 273.15, z).Output[0]
            h_0 = RP.REFPROPdll('CO2', 'PT', 'H', RP.MASS_BASE_SI, iMass, iFlag, p_0[i] * 1e6, tem[i] + 273.15, z).Output[0]
            d_0 = RP.REFPROPdll('CO2', 'PT', 'D', RP.MASS_BASE_SI, iMass, iFlag, p_0[i] * 1e6, tem[i] + 273.15, z).Output[0]
            h_c = RP.REFPROPdll('CO2', 'PS', 'H', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, s_0, z).Output[0]
            d_c = RP.REFPROPdll('CO2', 'PS', 'D', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, s_0, z).Output[0]
            G_cal = d_c * (2 * (h_0 - h_c)) ** 0.5
            s_l = RP.REFPROPdll('CO2', 'PQ', 'S', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, 0, z).Output[0]
            s_g = RP.REFPROPdll('CO2', 'PQ', 'S', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, 1, z).Output[0]
            x = (s_0 - s_l) / (s_g - s_l)

            if 0 < x < 1:
                miu_l = RP.REFPROPdll('CO2', 'PQ', 'VIS', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, 0, z).Output[0]
                miu_g = RP.REFPROPdll('CO2', 'PQ', 'VIS', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, 1, z).Output[0]
                miu_c = (x / miu_g + (1 - x) / miu_l) ** (-1)
            else:
                miu_c = RP.REFPROPdll('CO2', 'PS', 'VIS', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, s_0, z).Output[0]

            dp_d = G_cal ** 2 / 2 / d_0 / (0.84 ** 2) / 1e6
            p_d = p_0[i] - dp_d
            d_d = RP.REFPROPdll('CO2', 'PS', 'D', RP.MASS_BASE_SI, iMass, iFlag, p_d * 1e6, s_0, z).Output[0]
            dp_a = G_cal ** 2 * (1 / d_c - 1 / d_d) / 1e6
            Re = G_cal * dia[i] * 0.001 / miu_c
            f = 0.316 * Re ** (-0.25)
            dp_f = f * ldr[i] * G_cal ** 2 / (d_d + d_c) / 1e6
            dp = p_0[i] - dp_d - dp_a - dp_f - p_c[i]
            w = 1000 * G_cal * torch.pi * (dia[i] * 0.001 / 2) ** 2
            dp = dp / (max_val_pre - min_val_pre)
            w = (w - min_val_mas) / (max_val_mas - min_val_mas)
            dp_list.append(dp)
            w_list.append(w)

        dp_list = torch.tensor(dp_list, dtype=torch.float32, requires_grad=True)
        w_list = torch.tensor(w_list, dtype=torch.float32, requires_grad=True)
        return dp_list, w_list


########################################################################################################################
# Adjustable parameters
input_size = 4
output_size = 1
batch_size = 1
num_train = 1760
hidden_size = 22
########################################################################################################################


device = torch.device("cpu")

# Set the REFPROP
RP = REFPROPFunctionLibrary('C:/Program Files (x86)/REFPROP')
iMass = 1  # 1 represents the mass basis
iFlag = 0  # 0 represents the standard calculation process
MASS_BASE_SI = RP.GETENUMdll(iFlag, "MASS BASE SI").iEnum  # Only REFPROP 10.0 can use this function

data_train = pd.read_csv('training&validation_PINN.csv')  # Load the data
data_test = pd.read_csv('background data.csv')  # Load the data

# Save a copy of the original data
x_raw = data_train.drop('mas', axis=1).values
G_exp_raw = data_train['mas'].values.reshape(-1, 1)
x_test_raw = data_test.drop('mas', axis=1).values
G_exp_test_raw = data_test['mas'].values.reshape(-1, 1)

x_train_raw = x_raw[:num_train]
G_exp_train_raw = G_exp_raw[:num_train]

scaler_x = MinMaxScaler()
x_train_scaled = scaler_x.fit_transform(x_train_raw)
x_test_scaled = scaler_x.transform(x_test_raw)
scaler_G = MinMaxScaler()
G_exp_train_scaled = scaler_G.fit_transform(G_exp_train_raw)
G_exp_test_scaled = scaler_G.transform(G_exp_test_raw)

# Convert to torch tensors
x_train = torch.from_numpy(x_train_scaled).float()
G_exp_train = torch.from_numpy(G_exp_train_scaled).float()

x_test = torch.from_numpy(x_test_scaled).float()
G_exp_test = torch.from_numpy(G_exp_test_scaled).float()

min_val_ldr = scaler_x.data_min_[0]
max_val_ldr = scaler_x.data_max_[0]
min_val_pre = scaler_x.data_min_[1]
max_val_pre = scaler_x.data_max_[1]
min_val_dia = scaler_x.data_min_[2]
max_val_dia = scaler_x.data_max_[2]
min_val_tem = scaler_x.data_min_[3]
max_val_tem = scaler_x.data_max_[3]
min_val_mas = scaler_G.data_min_
max_val_mas = scaler_G.data_max_

# Create a DataLoader object to pair input with output
train_dataset = torch.utils.data.TensorDataset(x_train, G_exp_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

test_dataset = torch.utils.data.TensorDataset(x_test, G_exp_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

model = torch.load('PINN_model_L2_40000.pth')

model.eval()
with torch.no_grad():
    relative_errors = []
    test_predictions = []
    test_true_values = []
    w_pre_list = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)
        ldr = batch_x[:, 0]
        pre = batch_x[:, 1]
        dia = batch_x[:, 2]
        tem = batch_x[:, 3]
        p_c_pred = model(batch_x[:, :4])
        w_true = batch_y
        ldr = ldr * (max_val_ldr - min_val_ldr) + min_val_ldr
        pre = pre * (max_val_pre - min_val_pre) + min_val_pre
        dia = dia * (max_val_dia - min_val_dia) + min_val_dia
        tem = tem * (max_val_tem - min_val_tem) + min_val_tem
        p_c_pred = p_c_pred * (max_val_pre - min_val_pre) + min_val_pre
        w_true = w_true * (max_val_mas - min_val_mas) + min_val_mas
        dp_val, w_pre = model.physics(ldr, pre, dia, tem, p_c_pred)
        w_pre = w_pre * (max_val_mas - min_val_mas) + min_val_mas
        w_pre_list.append(w_pre.cpu().numpy().flatten())
        test_predictions.append(w_pre.cpu().numpy().flatten())
        test_true_values.append(w_true.cpu().numpy().flatten())
        w_true = w_true.squeeze()
        relative_error = (w_pre - w_true) / w_true * 100
        relative_errors.append(relative_error.cpu().numpy().flatten())

    print(np.mean(np.concatenate(np.abs(relative_errors))))
    plt.figure(figsize=(10, 5))
    plt.plot(np.concatenate(test_true_values), marker='o', linestyle='-', color='g', label='True Values')
    plt.plot(np.concatenate(test_predictions), marker='o', linestyle='-', color='b', label='Predictions')
    plt.title('Test set: True value vs Predicted value')
    plt.xlabel('Sample Index')
    plt.ylabel('Mass Flow (kg/s)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    w_pre_array = np.concatenate(w_pre_list)
    w_pre_df = pd.DataFrame(w_pre_array, columns=['Predicted Mass Flow'])
    w_pre_df.to_csv('LHS_PINN-HEM.csv', index=False)