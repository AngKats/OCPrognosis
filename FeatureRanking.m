% The script FeatureRanking.m runs all compared ML methods. 
% Loads the input features Data_2years.mat
% Other inputs can be changed:
% - predicted parameter: use OS_2years for Overall Survival or PFS_2years for Progression-free survival 
% - selected features as predictors: {'Age','Age_cat','Dissemination_cat','PS (WHO)','Time of surgery_cat','Residual','(SCS)','SCS_cat','Disease score','Chemotherapy_cat'}
%
% Usage: FeatureRanking
%
% Copyright (c) 2020-2021, Angeliki Katsenou
% email: angeliki.katsenou@bristol.ac.uk
% email: akatsenou@gmail.com


clc; clf; clear; clear global; close all;

% input Parameters:
load Data_2years;

myColours=lines(5); %define colour palette for the figures


[idx2_PFS,scores2_PFS] = fscmrmr(feat_2years,PFS_2years);

figure(1)
bar(scores2_PFS(idx2_PFS),'FaceColor',myColours(1,:),'EdgeColor',myColours(1,:));
xticklabels(new_feat_labels(idx2_PFS))
xtickangle(45)
ylabel('Feature Importance')
set(gca,'FontSize',14)
set(gcf,'papersize',[14,9])
f=gca;
savefig('FiguresForPaper/PFS_2years_MRMR.fig');
exportgraphics(f,'FiguresForPaper/PFS_2years_MRMR.eps')
close all

[idx2_OS,scores2_OS] = fscmrmr(feat_2years,OS_2years);

figure(2)
bar(scores2_OS(idx2_OS),'FaceColor',myColours(2,:),'EdgeColor',myColours(2,:));
xticklabels(new_feat_labels(idx2_OS))
xtickangle(45)
ylabel('Feature Importance')
set(gca,'FontSize',14)
set(gcf,'papersize',[14,9])
f=gca;
savefig('FiguresForPaper/OS_2years_MRMR.fig');
exportgraphics(f,'FiguresForPaper/OS_2years_MRMR.eps')
close all


%%%%%%%%%%%%%%%%%%%%%%%

[idx2_PFS_chi2,scores2_PFS_chi2] = fscchi2(feat_2years,PFS_2years);
figure(3)
bar(scores2_PFS_chi2(idx2_PFS_chi2),'FaceColor',myColours(4,:),'EdgeColor',myColours(4,:));
xticklabels(new_feat_labels(idx2_PFS_chi2))
xtickangle(45)
ylabel('Feature Importance')
set(gca,'FontSize',14)
set(gcf,'papersize',[14,9])
f=gca;
savefig('FiguresForPaper/PFS_2years_chi2.fig');
exportgraphics(f,'FiguresForPaper/PFS_2years_chi2.eps')
close all

[idx2_OS_chi2,scores2_OS_chi2] = fscchi2(feat_2years,OS_2years);
figure(4) 
bar(scores2_OS_chi2(idx2_OS_chi2), 'FaceColor',myColours(5,:),'EdgeColor',myColours(5,:));
xticklabels(new_feat_labels(idx2_OS_chi2))
xtickangle(45)
ylabel('Feature Importance')
set(gca,'FontSize',14)
set(gcf,'papersize',[14,9])
f=gca;
savefig('FiguresForPaper/OS_2years_chi2.fig');
exportgraphics(f,'FiguresForPaper/OS_2years_chi2.eps')
close all

