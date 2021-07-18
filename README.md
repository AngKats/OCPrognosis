# 2-year Prognosis in Advanced Stage High Grade Serous Ovarian Cancer: Selecting Features for ML-based Prognosis Prediction

This repository contains the code used to run the experiments used in paper A. Laios et al., "2-year prognosis in advanced stage high grade serous ovarian cancer: Selecting features for ML-based prognosis prediction" submitted to Cancer Control.

## Requirements

* Operating system: Windows, Linux or macOS
* Matlab v2020a (probably compatible with older versions)
    
## Steps to run the code

1. Clone the repository
2. Open Matlab
3. For the Feature ranking: 
    - Run the FeatureRanking.m script file to execute the chi-squared and the MRMR methods;
    - Run the FeatureSelection_RegularizationMethods.m script to execute the Lasso and Elastic Nets Methods.
4. For the ML prognosis prediction models, run the run_allMLmodels.m script. 

  
## How to reference this software:
This software is free and anyone is freely licensed to use, copy, study, and change the software in any way.
We would be grateful if the following works are cited in case of any publication that has used this software:

- Laios A, Katsenou A, Tan Y, et al 332 Two-year prognosis estimation of advanced high grade serous ovarian cancer patients using machine learning International Journal of Gynecologic Cancer 2020;30:A67-A68.



[//]: ## Acknowledgements
[//]: We acknowledge that part of this work was funded by the ???
