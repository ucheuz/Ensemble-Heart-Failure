# Ensemble-Heart-Failure
A machine learning ensemble model to diagnose heart failure severity by combining decision tree and logistic regression classifiers. Using bagging and boosting techniques, the model classifies patients into three categories: healthy (Label 0), mild heart failure (Label 1), and severe heart failure (Label 2). The ensemble approach enhances prediction accuracy and robustness compared to individual classifiers.

# Description
In this project, I aimed to enhance heart failure classification using ensemble learning techniques. I experimented with decision trees, logistic regression, bagging, and boosting to improve model performance. By implementing a BaggingClassifier with 20 estimators and bootstrap sampling, I increased model robustness compared to a single decision tree. Additionally, I applied AdaBoost with logistic regression as the base estimator to refine classification boundaries. The ensemble models demonstrated improved accuracy and generalization, highlighting the benefits of combining multiple weak learners. The final implementation included performance evaluation, comparison of classifiers, and analysis of why ensemble methods outperformed individual models.

# How to run
Option 1: Running the Python Script: Can be done directly in terminal. This will train the model and output predictions and accuracy scores.

Option 2: Running the Juypter Notebook: Download and open Ensemble Juypter Notebook.ipynb and run the cells in Task 3 step by step.

# Dataset
File: datasets/heart_failure_data_complete.csv
Description: A collection of patient records with features like HF (heart failure) labe, Ejection Fraction (EF), Global Longitudinal Strain (GLS) and QRS Complex (QRS)

# Results
Accuracy: 0.82
Conclusion: The AdaBoost classifier shows lower accuracy (0.82) compared to the Logistic Regression classifier (0.89). While AdaBoost improved recall for the first class (0.98), it significantly reduced recall for the second class (0.6). This is likely due to AdaBoost focusing more on harder-to-classify instances, which can lead to better performance for certain classes but worse overall results.

# Contributors
Kosiasochukwu Uchemudi Uzoka - Author
