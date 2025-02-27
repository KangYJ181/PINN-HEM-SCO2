import shap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = (1.8691 - 0.06231) / 2 * torch.sin(out) - (1.8691 + 0.06231) / 2
        out = out * (max_val_pre - min_val_pre) + min_val_pre
        return out


def physics(ldr, p_0, dia, tem, p_c):
    w_list = []
    z = [1.0]

    for i in range(p_0.size(0)):
        s_0 = RP.REFPROPdll('CO2', 'PT', 'S', RP.MASS_BASE_SI, iMass, iFlag, p_0[i] * 1e6, tem[i] + 273.15, z).Output[0]
        h_0 = RP.REFPROPdll('CO2', 'PT', 'H', RP.MASS_BASE_SI, iMass, iFlag, p_0[i] * 1e6, tem[i] + 273.15, z).Output[0]
        h_c = RP.REFPROPdll('CO2', 'PS', 'H', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, s_0, z).Output[0]
        d_c = RP.REFPROPdll('CO2', 'PS', 'D', RP.MASS_BASE_SI, iMass, iFlag, p_c[i] * 1e6, s_0, z).Output[0]
        G_cal = d_c * (2 * (h_0 - h_c)) ** 0.5
        w = 1000 * G_cal * torch.pi * (dia[i] * 0.001 / 2) ** 2
        w_list.append(w)

    return torch.stack(w_list)


def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        p_c = model(x_tensor)
        ldr = x_tensor[:, 0] * (max_val_ldr - min_val_ldr) + min_val_ldr
        pre = x_tensor[:, 1] * (max_val_pre - min_val_pre) + min_val_pre
        dia = x_tensor[:, 2] * (max_val_dia - min_val_dia) + min_val_dia
        tem = x_tensor[:, 3] * (max_val_tem - min_val_tem) + min_val_tem
        w_list = physics(ldr, pre, dia, tem, p_c)

    return w_list.numpy()

# Set the REFPROP
RP = REFPROPFunctionLibrary('C:/Program Files (x86)/REFPROP')
iMass = 1  # 1 represents the mass basis
iFlag = 0  # 0 represents the standard calculation process
MASS_BASE_SI = RP.GETENUMdll(iFlag, "MASS BASE SI").iEnum  # Only REFPROP 10.0 can use this function

data_train = pd.read_csv('training data.csv')
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
G_exp_train = torch.tensor(G_exp_train_scaled, dtype=torch.float32)

x_test = torch.tensor(x_test_scaled, dtype=torch.float32)

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

model = torch.load('-6_0.1.pth')

explainer = shap.KernelExplainer(model_predict, x_BGD_scaled)
shap_values = explainer.shap_values(x_test_scaled[0:10000],nsamples = 200)
shap_values_array = shap_values.reshape(10000, 4)
shap_df = pd.DataFrame(shap_values_array, columns=[f"Feature_{i+1}" for i in range(shap_values_array.shape[1])])
shap_df.to_csv('-6_0.1.csv', index=False)