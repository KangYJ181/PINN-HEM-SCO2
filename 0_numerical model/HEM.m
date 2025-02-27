clear;
clc

data = csvread('total data.csv', 1, 0);
[num, ~] = size(data);  % the number of input data

for n = 1 : num
% input parameters
ldr = data(n, 1);  % length to diameter ratio
pre = data(n, 2);  % pressure (MPa)
dia = data(n, 3);  % diameter (mm)
tem = data(n, 4) + 273.15;  % temperature (K)
mas = data(n,5);  % mass flow rate (g/s)

L = ldr .* dia;  % length of pipe
A = 0.25 * pi * (0.001 * dia) .^ 2;  % flow area (m^2)

% set iteration parameters
dpc = -0.001;  % step size of output pressure (MPa)
epsilon = 0.001;  % accuracy (MPa)
C = 0.84;  % form loss coefficient

% calculate initial properties
h0 = refpropm('H', 'T', tem, 'P', 1000 * pre, 'CO2');  % enthalpy (J/kg)
s0 = refpropm('S', 'T', tem, 'P', 1000 * pre, 'CO2');  % Entropy [J/(kg K)]
d0 = refpropm('D', 'T', tem, 'P', 1000 * pre, 'CO2');  % density (kg/m^3)

% iteration
pc = 7.3; % initial guess of output pressure (MPa)
pc_p = pc - epsilon;

while(abs(pc - pc_p) + 1e-6 >= epsilon)
    pc = pc + dpc;
    hc = refpropm('H', 'P', 1000 * pc, 'S', s0, 'CO2');
    dc = refpropm('D', 'P', 1000 * pc, 'S', s0, 'CO2');
    w = sqrt(2 * (h0 - hc));
    G = w * dc;  % mass flow [kg/(m^2 s)]

    if(pc <= 7.377 + 1e-6)
        sl = refpropm('S', 'P', 1000 * pc, 'Q', 0, 'CO2');
        sg = refpropm('S', 'P', 1000 * pc, 'Q', 1, 'CO2');
        x = (s0 - sl) / (sg - sl);  % mass quality
        if(x > 0 && x < 1)
            ul = refpropm('V', 'P', 1000 * pc, 'Q', 0, 'CO2');
            ug = refpropm('V', 'P', 1000 * pc, 'Q', 1, 'CO2');
            uc = (x / ug + (1 - x) / ul) ^ (-1);
        else
            uc = refpropm('V', 'P', 1000 * pc, 'S', s0, 'CO2');
        end
    else
        uc = refpropm('V', 'P', 1000 * pc, 'S', s0, 'CO2');
    end

    dp_c = G ^ 2 / d0 / (2 * C ^ 2) / 1e6;  % form drag pressure drop (MPa)
    p_1 = pre - dp_c;

    % calculate the friction factor
    Re = G * dia * 0.001 / uc;
    f = 0.316 / Re ^ 0.25;
    d_1 = refpropm('D', 'P', 1000 * p_1, 'S', s0, 'CO2');
    dp_a = (G ^ 2 * (1 / dc - 1 / d_1)) / 1e6;  % acceleration pressure drop (MPa)
    dp_f = f / 2 * L / dia * G ^ 2 / (dc + d_1) *2 / 1e6;  %frictional pressure drop (MPa)
	dp = dp_c+ dp_a + dp_f;
    pc_p = pre - dp;
    W0 = 1000 * A * G;  % mass flow [g/(s)]

    if(pc <= pc_p)
        break
    end

end

error(n,:) = (W0 - mas) / mas * 100;
error_ABS(n,:) = abs((W0 - mas) / mas * 100);
W(n, :) = W0;
p_c(n,:) = pc; 

fprintf('Prediction error of case %d: %.2f%%\n', n, (W0 - mas) / mas * 100)

end