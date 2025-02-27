import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary


# Pseudo random, guaranteed repeatability
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define the PINN network
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

        # Ensure that the output is in the interval [1.5, 7.3] after the inverse normalization
        # The minimum pressure in training concentration is 7.5 MPa and the maximum wis 14.01 MPa
        out = (1.8691 - 0.06231) / 2 * torch.sin(out) - (1.8691 + 0.06231) / 2
        return out


    def physics(self, ldr, p_0, dia, tem, p_c):
        # p_: pressure, MPa
        # dia: diameter, mm
        # tem: temperature, ℃
        # s_: entropy, J/(kg K)
        # h_: enthalpy, J/kg
        # d_: density, kg/m3
        # G_cal: calculated mass flow, kg/(m2 s)
        # dp_: pressure loss, MPa
        dp_list = []
        w_list = []
        z = [1.0]  # for pure fluid

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


def physics_constraints(model, p_c, batch_x):
    ldr = batch_x[:, 0]
    pre = batch_x[:, 1]
    dia = batch_x[:, 2]
    tem = batch_x[:, 3]
    ldr = ldr * (max_val_ldr - min_val_ldr) + min_val_ldr
    pre = pre * (max_val_pre - min_val_pre) + min_val_pre
    dia = dia * (max_val_dia - min_val_dia) + min_val_dia
    tem = tem * (max_val_tem - min_val_tem) + min_val_tem
    dp, w = model.physics(ldr, pre, dia, tem, p_c)
    return dp, w


def numerical_gradient(loss_fn, model, w_exp, batch_x, p_c):
    epsilon = 1e-6
    p_c_plus = p_c + epsilon
    loss_plus = loss_fn(model, w_exp, batch_x, p_c_plus)
    p_c_minus = p_c - epsilon
    loss_minus = loss_fn(model, w_exp, batch_x, p_c_minus)
    gradient = (loss_plus - loss_minus) / (2 * epsilon)
    return gradient


def custom_loss(model, w_exp, batch_x, p_c):
    x_raw = batch_x[:, :4]
    dp, w = physics_constraints(model, p_c, x_raw)
    dp = dp.unsqueeze(1)
    w = w.unsqueeze(1)
    loss_1 = nn.MSELoss()(dp, torch.zeros_like(dp))
    if torch.any(w_exp != 0):
        mask = w_exp != 0
        w_exp_ = w_exp[mask]
        w_ = w[mask]
        loss_2 = nn.MSELoss()(w_, w_exp_)
    else:
        loss_2 = torch.tensor(0.0, device=w.device)
    total_loss = weight * loss_1 + (1-weight) * loss_2
    return total_loss


def train_and_evaluate_PINN(learning_rate, hidden_size, weight_decay):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
    model = PINN(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load('data_driven_model_weights.pth'))
    #model = torch.load('PINN_model_L2_10000end.pth')
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_history = []

    # Initialize list to store the results for each epoch
    epoch_results = []

    for epoch in range(epochs):
        epoch_loss = 0
        # Training phase
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            optimizer.zero_grad()
            p_c = model(batch_x[:, :4])
            p_c = p_c * (max_val_pre - min_val_pre) + min_val_pre
            p_c.retain_grad()
            loss = custom_loss(model, batch_y, batch_x, p_c)
            grad_loss_p_c = numerical_gradient(custom_loss, model, batch_y, batch_x, p_c)
            grads = torch.autograd.grad(outputs=p_c, inputs=model.parameters(), grad_outputs=torch.ones_like(p_c),
                                        create_graph=True)

            for param, grad in zip(model.parameters(), grads):
                if grad is not None:
                    final_grad = grad * grad_loss_p_c
                    param.grad = final_grad
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Calculate relative error for validation set
        with torch.no_grad():
            relative_errors2 = []
            valid_predictions2 = []
            valid_true_values2 = []
            for i, (batch_x, batch_y) in enumerate(valid_loader):
                batch_x = batch_x.to(device)
                ldr = batch_x[:, 0]
                pre = batch_x[:, 1]
                dia = batch_x[:, 2]
                tem = batch_x[:, 3]
                p_c_pred2 = model(batch_x[:, :4])
                w_true2 = batch_y
                ldr = ldr * (max_val_ldr - min_val_ldr) + min_val_ldr
                pre = pre * (max_val_pre - min_val_pre) + min_val_pre
                dia = dia * (max_val_dia - min_val_dia) + min_val_dia
                tem = tem * (max_val_tem - min_val_tem) + min_val_tem
                p_c_pred2 = p_c_pred2 * (max_val_pre - min_val_pre) + min_val_pre
                w_true2 = w_true2 * (max_val_mas - min_val_mas) + min_val_mas
                dp_val, w_pre2 = model.physics(ldr, pre, dia, tem, p_c_pred2)
                w_pre2 = w_pre2 * (max_val_mas - min_val_mas) + min_val_mas
                valid_predictions2.append(w_pre2.cpu().numpy().flatten())
                valid_true_values2.append(w_true2.cpu().numpy().flatten())
                w_true2 = w_true2.squeeze()
                relative_error2 = (w_pre2 - w_true2) / w_true2 * 100
                relative_errors2.append(relative_error2.cpu().numpy().flatten())

            avg_relative_error2 = np.mean(np.concatenate(np.abs(relative_errors2)))

        # Print loss and relative errors
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
              f'Validation Relative Error: {avg_relative_error2:.2f}%')

        # Store the results for this epoch
        epoch_results.append({
            'Epoch': epoch + 1,
            'Loss': avg_loss,
            'Validation Relative Error': avg_relative_error2
        })

        # Save the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            torch.save(model, f'PINN_model_L2_{epoch + 1}.pth')  # Save the model at this epoch

    # Save the results to a CSV file after training is complete
    epoch_results_df = pd.DataFrame(epoch_results)
    epoch_results_df.to_csv('epoch_results.csv', index=False)  # Save as CSV file

    '''
    # Save the final model
    torch.save(model, 'PINN_model_final.pth')
    '''


# Adjustable parameters
########################################################################################################################
input_size = 4
output_size = 1
epochs = 40000
batch_size = 256
num_train = 1760
learning_rate = 0.0001
hidden_size = 22
weight_decay = 1e-6
weight = 0.1
########################################################################################################################


seed = 42
set_seed(seed)
device = torch.device("cpu")

# Set the REFPROP
RP = REFPROPFunctionLibrary('C:/Program Files (x86)/REFPROP')
iMass = 1  # 1 represents the mass basis
iFlag = 0  # 0 represents the standard calculation process
MASS_BASE_SI = RP.GETENUMdll(iFlag, "MASS BASE SI").iEnum  # Only REFPROP 10.0 can use this function

data = pd.read_csv('training&validation_PINN.csv')  # Load the data
x_raw = data.drop('mas', axis=1).values
G_exp_raw = data['mas'].values.reshape(-1, 1)

# Split the data into training and validation sets
x_train_raw = x_raw[:num_train]
G_exp_train_raw = G_exp_raw[:num_train]
x_val_raw = x_raw[num_train:]
G_exp_val_raw = G_exp_raw[num_train:]

# Data preprocessing: scale the training data to the 0~1 range
scaler_x = MinMaxScaler()
x_train_scaled = scaler_x.fit_transform(x_train_raw)
x_val_scaled = scaler_x.transform(x_val_raw)

# Inverse transformation parameters for the training data
min_val_ldr = scaler_x.data_min_[0]
max_val_ldr = scaler_x.data_max_[0]
min_val_pre = scaler_x.data_min_[1]
max_val_pre = scaler_x.data_max_[1]
min_val_dia = scaler_x.data_min_[2]
max_val_dia = scaler_x.data_max_[2]
min_val_tem = scaler_x.data_min_[3]
max_val_tem = scaler_x.data_max_[3]

scaler_G = MinMaxScaler()
G_exp_train_scaled = scaler_G.fit_transform(G_exp_train_raw)
G_exp_val_scaled = scaler_G.transform(G_exp_val_raw)
min_val_mas = scaler_G.data_min_
max_val_mas = scaler_G.data_max_

# Convert to torch tensors
x_train = torch.from_numpy(x_train_scaled).float()
G_exp_train = torch.from_numpy(G_exp_train_scaled).float()
x_val = torch.from_numpy(x_val_scaled).float()
G_exp_val = torch.from_numpy(G_exp_val_scaled).float()

# Create a DataLoader object to pair input with output
train_dataset = torch.utils.data.TensorDataset(x_train, G_exp_train)
val_dataset = torch.utils.data.TensorDataset(x_val, G_exp_val)
train_and_evaluate_PINN(learning_rate, hidden_size, weight_decay)