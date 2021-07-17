% The script run_allMLmodels.m runs all compared ML methods. 
% Loads the input features Data_2years.mat
% Other inputs can be changed:
% - number of iterations: NoIter
% - number of folds: NoFolds
% - predicted parameter: use OS_2years for Overall Survival or PFS_2years for Progression-free survival 
% - selected features as predictors: {'Age','Age_cat','Dissemination_cat','PS (WHO)','Time of surgery_cat','Residual','(SCS)','SCS_cat','Disease score','Chemotherapy_cat'}
%
% Usage: run_allMLmodels
%
% Copyright (c) 2020-2021, Angeliki Katsenou
% email: angeliki.katsenou@bristol.ac.uk
% email: akatsenou@gmail.com


clc; clf; clear; clear global; close all;

addpath 'MLmodels'

%% Initialising input parameters
load Data_2years.mat; % matfile with all input features
NoIter=100; % Define the number of iterations
NoFolds=10; % Define the number of folds
inputFeatures=[feat_2years OS_2years]; % Define the predicted parameter: use OS_2years for Overall Survival or PFS_2years for Progression-free survival 
predictorNames = {'column_4', 'column_5', 'column_6', 'column_9', 'column_10'}; % {'Age','Age_cat','Dissemination_cat','PS (WHO)','Time of surgery_cat','Residual','(SCS)','SCS_cat','Disease score','Chemotherapy_cat'}

%% Run Models

for i=1:NoIter
    
     display('-------- Run all Models ----------')
     display(['Iteration: ' int2str(i)])      
     [AccuracySVMquadr2(i,1), validationPredictionsSVMquadr2(:,i)] = trainSVMquadr_2(inputFeatures, NoFolds, predictorNames);
     [AccuracySVMcubic2(i,1), validationPredictionsSVMcubic2(:,i)] = trainSVMcubic(inputFeatures, NoFolds, predictorNames);
     [AccuracyLogisticRegression2(i,1), validationPredictionsLogisticRegression2(:,i)] = trainLogisticRegression(inputFeatures, NoFolds, predictorNames);      
     [AccuracyNaiveBayes2(i,1), validationPredictionsNaiveBayes2(:,i)] = NaiveBayes(inputFeatures, NoFolds, predictorNames);
     [AccuracyWKNNs52(i,1), validationPredictionsWKNNs52(:,i)] = trainWeightedKNN(inputFeatures, 5, NoFolds,predictorNames);
     [AccuracyWKNNs102(i,1), validationPredictionsWKNNs102(:,i)] = trainWeightedKNN(inputFeatures, 10, NoFolds,predictorNames)  ;
     [AccuracyEnsmble2(i,1), validationPredictionsEnsmble2(:,i)] = trainEnsemble(inputFeatures, NoFolds, predictorNames);
     [AccuracyEnsmbleSubd2(i,1), validationPredictionsEnsmbleSubd2(:,i)] = EnsembleTreesSubdiscrim(inputFeatures, NoFolds, predictorNames);


    display('-------- SVMquadr model ----------')

    [~,~,~,AUC_0_SVMquadr2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsSVMquadr2(:,i),'0');
    [~,~,~,AUC_1_SVMquadr2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsSVMquadr2(:,i),'1');

    cmSVMquadr=confusionchart(inputFeatures(:,end), validationPredictionsSVMquadr2(:,i));
    TPSVMquadr=cmSVMquadr.NormalizedValues(1,1);
    TNSVMquadr=cmSVMquadr.NormalizedValues(2,2);
    FPSVMquadr=cmSVMquadr.NormalizedValues(2,1);
    FNSVMquadr=cmSVMquadr.NormalizedValues(1,2);

    precisionSVMquadr2(i,1)=TPSVMquadr/(TPSVMquadr+FPSVMquadr);
    recallSVMquadr2(i,1)=TPSVMquadr/(TPSVMquadr+FNSVMquadr);
    specificitySVMquadr2(i,1)=TNSVMquadr/(TNSVMquadr+FPSVMquadr);
    fscoreSVMquadr2(i,1)=2*precisionSVMquadr2(i,1)*recallSVMquadr2(i,1)/(precisionSVMquadr2(i,1)+recallSVMquadr2(i,1));
    gscoreSVMquadr2(i,1)=sqrt(precisionSVMquadr2(i,1)*recallSVMquadr2(i,1));

    display('-------- SVMcubic model ----------')

    [~,~,~,AUC_0_SVMcubic2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsSVMcubic2(:,i),'0');
    [~,~,~,AUC_1_SVMcubic2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsSVMcubic2(:,i),'1');

    cmSVMcubic=confusionchart(inputFeatures(:,end), validationPredictionsSVMcubic2(:,i));
    TPSVMcubic=cmSVMcubic.NormalizedValues(1,1);
    TNSVMcubic=cmSVMcubic.NormalizedValues(2,2);
    FPSVMcubic=cmSVMcubic.NormalizedValues(1,2);
    FNSVMcubic=cmSVMcubic.NormalizedValues(2,1);

    precisionSVMcubic2(i,1)=TPSVMcubic/(TPSVMcubic+FPSVMcubic);
    recallSVMcubic2(i,1)=TPSVMcubic/(TPSVMcubic+FNSVMcubic);
    specificitySVMcubic2(i,1)=TNSVMcubic/(TNSVMcubic+FPSVMcubic);
    fscoreSVMcubic2(i,1)=2*precisionSVMcubic2(i,1)*recallSVMcubic2(i,1)/(precisionSVMcubic2(i,1)+recallSVMcubic2(i,1));
    gscoreSVMcubic2(i,1)=sqrt(precisionSVMcubic2(i,1)*recallSVMcubic2(i,1));

    display('-------- LogisticRegression model ----------')

    [~,~,~,AUC_0_LogisticRegression2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsLogisticRegression2(:,i),'0');
    [~,~,~,AUC_1_LogisticRegression2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsLogisticRegression2(:,i),'1');

    cmLogisticRegression=confusionchart(inputFeatures(:,end), validationPredictionsLogisticRegression2(:,i));
    TPLogisticRegression=cmLogisticRegression.NormalizedValues(1,1);
    TNLogisticRegression=cmLogisticRegression.NormalizedValues(2,2);
    FPLogisticRegression=cmLogisticRegression.NormalizedValues(2,1);
    FNLogisticRegression=cmLogisticRegression.NormalizedValues(1,2);

    precisionLogisticRegression2(i,1)=TPLogisticRegression/(TPLogisticRegression+FPLogisticRegression);
    recallLogisticRegression2(i,1)=TPLogisticRegression/(TPLogisticRegression+FNLogisticRegression);
    specificityLogisticRegression2(i,1)=TNLogisticRegression/(TNLogisticRegression+FPLogisticRegression);
    fscoreLogisticRegression2(i,1)=2*precisionLogisticRegression2(i,1)*recallLogisticRegression2(i,1)/(precisionLogisticRegression2(i,1)+recallLogisticRegression2(i,1));
    gscoreLogisticRegression2(i,1)=sqrt(precisionLogisticRegression2(i,1)*recallLogisticRegression2(i,1));

     display('-------- NaiveBayes model ----------')

    [~,~,~,AUC_0_NaiveBayes2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsNaiveBayes2(:,i),'0');
    [~,~,~,AUC_1_NaiveBayes2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsNaiveBayes2(:,i),'1');

    cmNaiveBayes=confusionchart(inputFeatures(:,end), validationPredictionsNaiveBayes2(:,i));
    TPNaiveBayes=cmNaiveBayes.NormalizedValues(1,1);
    TNNaiveBayes=cmNaiveBayes.NormalizedValues(2,2);
    FPNaiveBayes=cmNaiveBayes.NormalizedValues(2,1);
    FNNaiveBayes=cmNaiveBayes.NormalizedValues(1,2);

    precisionNaiveBayes2(i,1)=TPNaiveBayes/(TPNaiveBayes+FPNaiveBayes);
    recallNaiveBayes2(i,1)=TPNaiveBayes/(TPNaiveBayes+FNNaiveBayes);
    specificityNaiveBayes2(i,1)=TNNaiveBayes/(TNNaiveBayes+FPNaiveBayes);
    fscoreNaiveBayes2(i,1)=2*precisionNaiveBayes2(i,1)*recallNaiveBayes2(i,1)/(precisionNaiveBayes2(i,1)+recallNaiveBayes2(i,1));
    gscoreNaiveBayes2(i,1)=sqrt(precisionNaiveBayes2(i,1)*recallNaiveBayes2(i,1));

    display('-------- WKNNs5 model ----------')

    [~,~,~,AUC_0_WKNNs52(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsWKNNs52(:,i),'0');
    [~,~,~,AUC_1_WKNNs52(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsWKNNs52(:,i),'1');

    cmWKNNs5=confusionchart(inputFeatures(:,end), validationPredictionsWKNNs52(:,i));
    TPWKNNs5=cmWKNNs5.NormalizedValues(1,1);
    TNWKNNs5=cmWKNNs5.NormalizedValues(2,2);
    FPWKNNs5=cmWKNNs5.NormalizedValues(2,1);
    FNWKNNs5=cmWKNNs5.NormalizedValues(1,2);

    precisionWKNNs52(i,1)=TPWKNNs5/(TPWKNNs5+FPWKNNs5);
    recallWKNNs52(i,1)=TPWKNNs5/(TPWKNNs5+FNWKNNs5);
    specificityWKNNs52(i,1)=TNWKNNs5/(TNWKNNs5+FPWKNNs5);
    fscoreWKNNs52(i,1)=2*precisionWKNNs52(i,1)*recallWKNNs52(i,1)/(precisionWKNNs52(i,1)+recallWKNNs52(i,1));
    gscoreWKNNs52(i,1)=sqrt(precisionWKNNs52(i,1)*recallWKNNs52(i,1));

    display('-------- WKNNs10 model ----------')

    [~,~,~,AUC_0_WKNNs102(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsWKNNs102(:,i),'0');
    [~,~,~,AUC_1_WKNNs102(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsWKNNs102(:,i),'1');

    cmWKNNs10=confusionchart(inputFeatures(:,end), validationPredictionsWKNNs102(:,i));
    TPWKNNs10=cmWKNNs10.NormalizedValues(1,1);
    TNWKNNs10=cmWKNNs10.NormalizedValues(2,2);
    FPWKNNs10=cmWKNNs10.NormalizedValues(2,1);
    FNWKNNs10=cmWKNNs10.NormalizedValues(1,2);

    precisionWKNNs102(i,1)=TPWKNNs10/(TPWKNNs10+FPWKNNs10);
    recallWKNNs102(i,1)=TPWKNNs10/(TPWKNNs10+FNWKNNs10);
    specificityWKNNs102(i,1)=TNWKNNs10/(TNWKNNs10+FPWKNNs10);
    fscoreWKNNs102(i,1)=2*precisionWKNNs102(i,1)*recallWKNNs102(i,1)/(precisionWKNNs102(i,1)+recallWKNNs102(i,1));
    gscoreWKNNs102(i,1)=sqrt(precisionWKNNs102(i,1)*recallWKNNs102(i,1));

    display('-------- Ensemble model ----------')

    [~,~,~,AUC_0_Ensmble2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsEnsmble2(:,i),'0');
    [~,~,~,AUC_1_Ensmble2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsEnsmble2(:,i),'1');

    cmEnsmble=confusionchart(inputFeatures(:,end), validationPredictionsEnsmble2(:,i));
    TPEnsmble=cmEnsmble.NormalizedValues(1,1);
    TNEnsmble=cmEnsmble.NormalizedValues(2,2);
    FPEnsmble=cmEnsmble.NormalizedValues(2,1);
    FNEnsmble=cmEnsmble.NormalizedValues(1,2);

    precisionEnsmble2(i,1)=TPEnsmble/(TPEnsmble+FPEnsmble);
    recallEnsmble2(i,1)=TPEnsmble/(TPEnsmble+FNEnsmble);
    specificityEnsmble2(i,1)=TNEnsmble/(TNEnsmble+FPEnsmble);
    fscoreEnsmble2(i,1)=2*precisionEnsmble2(i,1)*recallEnsmble2(i,1)/(precisionEnsmble2(i,1)+recallEnsmble2(i,1));
    gscoreEnsmble2(i,1)=sqrt(precisionEnsmble2(i,1)*recallEnsmble2(i,1));


    display('-------- EnsembleSubd model ----------')

    [~,~,~,AUC_0_EnsmbleSubd2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsEnsmbleSubd2(:,i),'0');
    [~,~,~,AUC_1_EnsmbleSubd2(i,1)] = perfcurve(inputFeatures(:,end),validationPredictionsEnsmbleSubd2(:,i),'1');

    cmEnsmbleSubd=confusionchart(inputFeatures(:,end), validationPredictionsEnsmbleSubd2(:,i));
    TPEnsmbleSubd=cmEnsmbleSubd.NormalizedValues(1,1);
    TNEnsmbleSubd=cmEnsmbleSubd.NormalizedValues(2,2);
    FPEnsmbleSubd=cmEnsmbleSubd.NormalizedValues(2,1);
    FNEnsmbleSubd=cmEnsmbleSubd.NormalizedValues(1,2);

    precisionEnsmbleSubd2(i,1)=TPEnsmbleSubd/(TPEnsmbleSubd+FPEnsmbleSubd);
    recallEnsmbleSubd2(i,1)=TPEnsmbleSubd/(TPEnsmbleSubd+FNEnsmbleSubd);
    specificityEnsmbleSubd2(i,1)=TNEnsmbleSubd/(TNEnsmbleSubd+FPEnsmbleSubd);
    fscoreEnsmbleSubd2(i,1)=2*precisionEnsmbleSubd2(i,1)*recallEnsmbleSubd2(i,1)/(precisionEnsmbleSubd2(i,1)+recallEnsmbleSubd2(i,1));
    gscoreEnsmbleSubd2(i,1)=sqrt(precisionEnsmbleSubd2(i,1)*recallEnsmbleSubd2(i,1));

end

display('-------- Mean Prediction Accuracy per Model after NoIter iterations --------')
mean_stats(1,:) = [mean(AccuracySVMquadr2(1:NoIter,1)) mean(AUC_1_SVMquadr2(1:NoIter,1)) mean(AUC_0_SVMquadr2(1:NoIter,1)) mean(precisionSVMquadr2(1:NoIter,1)) mean(recallSVMquadr2(1:NoIter,1)) mean(specificitySVMquadr2(1:NoIter,1)) mean(fscoreSVMquadr2(1:NoIter,1)) mean(gscoreSVMquadr2(1:NoIter,1))]
mean_stats(2,:) = [mean(AccuracySVMcubic2(1:NoIter,1)) mean(AUC_1_SVMcubic2(1:NoIter,1)) mean(AUC_0_SVMcubic2(1:NoIter,1)) mean(precisionSVMcubic2(1:NoIter,1)) mean(recallSVMcubic2(1:NoIter,1)) mean(specificitySVMcubic2(1:NoIter,1))  mean(fscoreSVMcubic2(1:NoIter,1)) mean(gscoreSVMcubic2(1:NoIter,1))]
mean_stats(3,:) = [mean(AccuracyLogisticRegression2(1:NoIter,1)) mean(AUC_1_LogisticRegression2(1:NoIter,1)) mean(AUC_0_LogisticRegression2(1:NoIter,1)) mean(precisionLogisticRegression2(1:NoIter,1)) mean(recallLogisticRegression2(1:NoIter,1)) mean(specificityLogisticRegression2(1:NoIter,1)) mean(fscoreLogisticRegression2(1:NoIter,1)) mean(gscoreLogisticRegression2(1:NoIter,1))]
mean_stats(4,:) = [mean(AccuracyNaiveBayes2(1:NoIter,1)) mean(AUC_1_NaiveBayes2(1:NoIter,1)) mean(AUC_0_NaiveBayes2(1:NoIter,1)) mean(precisionNaiveBayes2(1:NoIter,1)) mean(recallNaiveBayes2(1:NoIter,1)) mean(specificityNaiveBayes2(1:NoIter,1)) mean(fscoreNaiveBayes2(1:NoIter,1)) mean(gscoreNaiveBayes2(1:NoIter,1))]
mean_stats(5,:) = [mean(AccuracyWKNNs52(1:NoIter,1)) mean(AUC_1_WKNNs52(1:NoIter,1)) mean(AUC_0_WKNNs52(1:NoIter,1)) mean(precisionWKNNs52(1:NoIter,1)) mean(recallWKNNs52(1:NoIter,1)) mean(specificityWKNNs52(1:NoIter,1))  mean(fscoreWKNNs52(1:NoIter,1)) mean(gscoreWKNNs52(1:NoIter,1))]
mean_stats(6,:) = [mean(AccuracyWKNNs102(1:NoIter,1)) mean(AUC_1_WKNNs102(1:NoIter,1)) mean(AUC_0_WKNNs102(1:NoIter,1)) mean(precisionWKNNs102(1:NoIter,1)) mean(recallWKNNs102(1:NoIter,1)) mean(specificityWKNNs102(1:NoIter,1)) mean(fscoreWKNNs102(1:NoIter,1)) mean(gscoreWKNNs102(1:NoIter,1))]
mean_stats(7,:) = [mean(AccuracyEnsmble2(1:NoIter,1)) mean(AUC_1_Ensmble2(1:NoIter,1)) mean(AUC_0_Ensmble2(1:NoIter,1)) mean(precisionEnsmble2(1:NoIter,1)) mean(recallEnsmble2(1:NoIter,1)) mean(specificityEnsmble2(1:NoIter,1)) mean(fscoreEnsmble2(1:NoIter,1)) mean(gscoreEnsmble2(1:NoIter,1))]
mean_stats(8,:) = [mean(AccuracyEnsmbleSubd2(1:NoIter,1)) mean(AUC_1_EnsmbleSubd2(1:NoIter,1)) mean(AUC_0_EnsmbleSubd2(1:NoIter,1)) mean(precisionEnsmbleSubd2(1:NoIter,1)) mean(recallEnsmbleSubd2(1:NoIter,1)) mean(specificityEnsmbleSubd2(1:NoIter,1)) mean(fscoreEnsmbleSubd2(1:NoIter,1)) mean(gscoreEnsmbleSubd2(1:NoIter,1))]

