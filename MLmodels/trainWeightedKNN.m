function [validationAccuracy, validationPredictions] = trainWeightedKNN(trainingData,NoFolds, NoNeighbors, predictorNames)
% Usage 
%  [validationAccuracy, validationPredictions] = trainWeightedKNN(trainingData, NoFolds, NoNeighbors, predictorNames)
%
%  Input:
%      trainingData: The input data are organised in columns: features and groundtruth (last column)
%      NoFolds: A variable with the number of folds for corss-validation
%      NoNeighbors: A variable with the number of neighbors
%      
%  Output:
%      validationPredictions: A vector with the predicted values.
%      validationAccuracy: A double containing the accuracy in percent.
%
% Copyright (c) 2020-2021, Angeliki Katsenou
% email: angeliki.katsenou@bristol.ac.uk
% email: akatsenou@gmail.com



% Extract predictors and response
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11'});

predictors = inputTable(:, predictorNames);
response = inputTable.column_11;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

% Train a classifier
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', NoNeighbors, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationKNN = classificationKNN;

% Extract predictors and response

inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11'});
predictors = inputTable(:, predictorNames);
response = inputTable.column_11;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', NoFolds);

% Compute validation predictions
[validationPredictions, ~] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
