% The script FeatureSelection_RegularizationMethods.m runs all compared ML methods. 
% Loads the input features Data_2years.mat
% Other inputs can be changed:
% - number of folds: NoFolds
% - predicted parameter: use OS_2years for Overall Survival or PFS_2years for Progression-free survival 
% - selected features as predictors: {'Age','Age_cat','Dissemination_cat','PS (WHO)','Time of surgery_cat','Residual','(SCS)','SCS_cat','Disease score','Chemotherapy_cat'}
%
% Usage: FeatureSelection_RegularizationMethods
%
% Copyright (c) 2020-2021, Angeliki Katsenou
% email: angeliki.katsenou@bristol.ac.uk
% email: akatsenou@gmail.com


clear; close all;

% input Parameters:
load Data_2years;
features = feat_2years;
PredictedParameter = [PFS_2years OS_2years];
NoFolds = 10;

for i = 1:size(PredictedParameter,2)
    
    if i == 1
        param = 'PFS';
    else
        param = 'OS';
    end
        
    
    
    % Applying Lasso 
    fprintf(['--- Running Lasso for ' param '---\n'])

    [L, FitInfo_Lasso] = lasso(features, PredictedParameter(:,i),'CV', NoFolds);
    %[selFeat_Lasso] = FeatureLabels(find( L(:, FitInfo_Lasso.IndexMinMSE)~=0) ); % if the min MSE is to be used as a selection criterion
    
    lassoPlot(L, FitInfo_Lasso,'PlotType','Lambda','XScale','log', 'PredictorNames', FeatureLabels);
    %savefig(['lassoPlot_' param '.fig']) % to save the figure



    % Applying Elastic Nets
    fprintf(['--- Running Elastic Nets for ' param '---\n'])
    
    [ElN, FitInfo_ElNet] = lasso(features, PredictedParameter(:,i),'CV', NoFolds, 'Alpha', 0.75);
    %[selFeat_ElNet] = FeatureLabels( find( L(:,FitInfo_ElNet.IndexMinMSE)~=0) ); % if the min MSE is to be used as a selection criterion
    
    lassoPlot(ElN, FitInfo_ElNet,'PlotType','Lambda','XScale','log', 'PredictorNames', FeatureLabels);
    %savefig(['elnetPlot_' param '.fig']) % to save the figure

end