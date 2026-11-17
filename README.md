# Bank Customer Churn Prediction

**Neural-network classification | Python | Customer analytics**

## Overview
This project explores whether a bank customer is likely to leave, using a dataset described in the original notebook as containing **10,000 customer records**. It compares neural-network configurations and examines how class imbalance affects the ability to identify customers who churn.

## Problem
Missing customers who are likely to leave can limit a bank’s ability to act early. The project therefore prioritizes **recall for the churn class**, rather than optimizing accuracy alone.

## Workflow
1. Explore the data and remove identifier columns (`RowNumber`, `CustomerId`, `Surname`).
2. Prepare categorical and numerical features, split the dataset, and scale inputs.
3. Train feed-forward neural networks with **SGD** and **Adam** optimizers.
4. Compare dropout regularization and **SMOTE** oversampling on training data.
5. Evaluate classification results using confusion matrices and precision, recall, F1-score, accuracy and ROC-AUC.

## Model selection
The original notebook selects **SMOTE + Adam** as the preferred configuration, because identifying customers likely to churn (reducing false negatives) was the primary objective. The SMOTE + Adam + Dropout alternative was reported to have slightly higher ROC-AUC, but was not selected on the business-priority criterion.

No deployment or production integration is claimed.

## Tech stack
- **Language:** Python
- **Libraries:** NumPy, pandas, Matplotlib, seaborn, scikit-learn, imbalanced-learn, TensorFlow/Keras
- **Methods:** feed-forward neural networks, SGD, Adam, dropout, SMOTE, classification metrics

## Explore the work
- [Jupyter notebook](notebooks/bank_churn_prediction.ipynb) — extracted from the supplied HTML notebook, with code and explanatory markdown preserved.
- [Notebook figures](images/) — figures extracted from the supplied HTML export. These are provided as original outputs, not new performance claims.

## Run locally
1. Create a Python virtual environment and install dependencies: `pip install -r requirements.txt`.
2. Obtain the **authorized** `Churn.csv` dataset and place it in the working directory from which the notebook is run (or adjust the path in the data-loading cell).
3. Launch `jupyter notebook` and open `notebooks/bank_churn_prediction.ipynb`.

**Reproducibility note:** The dataset is **not included** because its redistribution rights have not been verified. Library versions are not pinned, and the notebook has not been re-executed in this environment. The HTML export is the source of the code; review the extracted notebook before publishing.

## Context
Prepared as a technical portfolio project. The problem statement and modeling approach are derived from the supplied coursework notebook. This repository does not claim that the bank deployed the model.
