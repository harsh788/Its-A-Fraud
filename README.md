# Its-A-Fraud
This repository contains the progress of the QuantumAI team for the ML project.

## Documentation
Steps on how to run the code:
1. Fork and clone this repository by running the command `git clone {url}` on your terminal.
2. Download the train and test data from [here](https://www.kaggle.com/competitions/its-a-fraud/data) and move it to your working directory.
3. Run the **ItsAFraud.ipynb** file after updating the path details of the recently downloaded .csv files. This will generate the preprocessed dataframe.
4. The preprocessed dataframe is already included in this repo for reference.
5. There are five models (_Logistic Regression, Naive Bayes, KNN, Random Forest Classifier, XGBoost_) which can be applied on the preprocessed dataframe for training the parameters. These models are present in their respective .py files
6. Running these models will generate the output .csv files which can be directly submitted on kaggle.
