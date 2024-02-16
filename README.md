# credit-risk-classification

## Overview of the Analysis

For this use case, I set out to use supervised machine learning to analyze loan performance on a relatively large dataset (approx 77k rows). The data did not have an overwhelming number of columns, so a simple logistical regression made sense for the analysis. 

The purpose of the analysis is to train a model to help a lender understand the risk associated with different borrower profiles. The data had a 'loan_status' column that flagged the data into two categories; low-risk and high-risk, based on historical performance. The value in this exercise is being able to predict the outcome of the loan_status column with a high degree of accuracy and precision. 

## Process

The first step of the process is to import the dataset from a csv file to a pandas dataframe. This is a simple way to analyze since dataframes are easily manipulated. As stated above, the loan_status column is the dependent variable so we'll separate it from rest of the dataframe.

After isolating the risk performance column (y) and turning it into the dependent variable, I used the other columns (loan_size, interest_rate, borrower_income, debt_to_income, number_of_accounts, derogatory_marks and total_debt) as features (independent variables). These features are typically part of any credit application and can be found on a standard credit report. 

I checked the balance of the loan_status column and about 3.3% of the loans are marked as high-risk. Next, the data was split into traning and testing datasets with the training set. This puts about 75% of the data into the training bucket. 

Next up, a logistic regression model is created using the sklearn.linear_model LogisiticRegression function. The requirements call for a randomstate=1 value so others are able to observe the same shape. Once this is instantiated, the model is fit/trained. Testing predictions are then generated, using the X_test dataset. The models (both train and test) were evaluated and both scored over 99%. 

A balanced accuracy score, confusion matrix and classification report are generated, next. 

To gain another perspective on the data, the process was repeated using a new library (RandomOverSampler) with resampled training data. This split the datasets into 50/50 buckets. 

## Results

Here are the analysis results of the two models: 

Original Training Data
* Accuracy Score (f1): 99%
* Precision: Low-risk: 100%, High-risk: 87% (weighted avg: 99%)
* Recall: Low-risk: 100%, High-risk: 95% (weighted avg: 99%)

Over Sampled Training Data
* Accuracy Score (f1): 100%
* Precision: Low-risk: 100%, High-risk: 87% (weighted avg: 100%)
* Recall: Low-risk: 100%, High-risk: 100% (weighted avg: 100%)

## Summary

In real-world applications, both models performed well. The over sampled model had a slight edge in its performance on the overall f1 scores. 

For loan default predictions, in general, it's important to pay attention to the precision of the model. You want to minimize false positives to avoid financial loss. The first model had 32 false positives for low-risk loans, whereas the overfit model had 2. The oversampled model predicted more high-risk loans. The bank would lose out on the revenue from these potential clients by denying their applications. 

To truly understand the financial value of the model, you'd need to look at the loan spreads and ask whether it's better to lend to more clients with a higher default rate or lend to less clients with lower defaults. A case could be made for both, depending on the size and per/loan profitability of the portfolio. 

