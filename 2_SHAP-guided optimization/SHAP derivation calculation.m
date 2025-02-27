clc,clear

% Extract validation errors in batches
% name1 = 'E:\文章写作\PINN-SCO2-2\1 code new\training\基于SHAP的优化\grid search\L2_1e-10\w_0.';
% name2 = '\epoch_results.csv';
% validation_error = [];
% for k = 6:9
%     name = [name1,num2str(k),name2];
%     validation_error = [validation_error;csvread(name,40000,2)];
% end

% Calculate SHAP slopes in batches
name1 = 'E:\文章写作\PINN-SCO2-2\1 code new\training\基于SHAP的优化\SHAP calculation\L2_1e-8\-8_0.';
% Ten equal dispersion points of ldr and tem
ldr = [1,19.56,38.11,56.67,75.22,93.78,112.33,130.89,149.44,168];
tem = [34,85.89,137.78,189.67,241.56,293.44,345.33,397.22,449.11,501];
% Add interpolation points, spaced 0.01 apart
ldrq = round((1:0.01:168)*100)/100;
temq = round((34:0.01:501)*100)/100;
dSHAP_dx = [];  % Store the result of calculating the average derivative of each case
SHAP_ldr_total = zeros(10,9);  % Store SHAP curves for all cases
SHAP_tem_total = zeros(10,9);
for k1 = 1:1
    name2 = '.csv';
    name = [name1,num2str(k1),name2];
    ave_SHAP_ldr = [];
    ave_SHAP_tem = [];

    % The SHAP average values of ldr and tem under each scatter point in this case were extracted
    for k2 = 1:10
        SHAP_ldr = csvread(name,(k2-1)*1000+1,0,[(k2-1)*1000+1,0,k2*1000,0]);
        SHAP_tem = csvread(name,k2,3,[k2,3,9990+k2,3]);
        SHAP_tem = SHAP_tem(1:10:end);
        ave_SHAP_ldr = [ave_SHAP_ldr;mean(SHAP_ldr)];
        ave_SHAP_tem = [ave_SHAP_tem;mean(SHAP_tem)];
    end

    SHAP_ldr_total(:,k1) = ave_SHAP_ldr;
    SHAP_tem_total(:,k1) = ave_SHAP_tem;
    % Set the starting point SHAP of each case to 1
    ave_SHAP_ldr = ave_SHAP_ldr / ave_SHAP_ldr(1);
    ave_SHAP_tem = ave_SHAP_tem / ave_SHAP_tem(1);
    % Added point interpolation, ready for derivative
    ave_SHAP_ldrq = interp1(ldr,ave_SHAP_ldr,ldrq,'spline');
    ave_SHAP_temq = interp1(tem,ave_SHAP_tem,temq,'spline');
    % Calculate the derivatives at all interpolation points
    dSHAP_dldr = gradient(ave_SHAP_ldrq,ldrq);
    dSHAP_dtem = gradient(ave_SHAP_temq,temq);
    % Only the derivatives at the test hubs are extracted
    [~,index_ldr] = ismember(ldr(4:end),ldrq);
    [~,index_tem] = ismember(tem(3:end),temq);
    dSHAP_dx = [dSHAP_dx;[mean(dSHAP_dldr(index_ldr)),mean(dSHAP_dtem(index_tem))]];
end