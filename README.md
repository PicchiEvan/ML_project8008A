# ML_project8008A
## File structure
The files you find at the **root** are pluto notebooks (and 2 csv) :
- First_try.jl : contains Logistic regression, (with normalization, L1,L2, ... ) And KNN classification
- Neural_Net.jl : Contains Neural network classifier, and tree methods Random forest and XGBoost (tree)
- Data_vis_pca.jl: Contains the data visualisation and preprocessing (in this file you can also edit the seed for the separation of the "subtraining" and "subtesting" sets) and PCA biplot
- ResultBestLinear.csv : best result submitted on kaggle for linear classification
- ResultXGC.csv : Best result submitted on kaggle for XGBoostClassification
- Machine_learning_Project.pdf : report

In the **Data/** folder you will find :
- test_data_nv.csv : the test set used to estimate the test error
- train_data_nv.csv the set used for hyperparameter tuning and training of the different set
- trainingdata.csv : training data found on kaggle
- testdata.csv : test data found on kaggle

In the **Script/** folder you will find different scripts used to find the optimal parameters for the different models used in this project
- Logistic.jl : contains hyperparameters optimisation for logistic classification (with L1, L2, L1 + standardization, L2+standardization)
- NeuralNetwork.jl:contains hyperparameters optimisation for Neural network
- RandomTree.jl : contains hyperparameters optimisation (n_tree) for RandomTree method
- XGBoost.jl : contains hyperparameters optimisation for XGBoostClassifier
## Version of Julia : 1.6.2 (the version of the MLCourse environment are the same as the start of the course)
## Version of pluto : v0.15.1
