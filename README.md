📞 Telco Customer Churn Prediction
📌 Project Overview

This project focuses on predicting customer churn (whether a customer will leave the company or not) for a telecom dataset.
The goal is to analyze customer behavior, train different machine learning models, and deploy a Streamlit web app for real-time predictions.

📂 Dataset

Two datasets were provided:

Training Dataset: customer_churn_dataset-training-master.csv

Testing Dataset: customer_churn_dataset-testing-master.csv

Features Included:

Demographic Information: Age, Gender

Service Information: Subscription Type, Contract Length

Usage Behavior: Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction

Target Variable: Churn (1 = customer churned, 0 = customer stayed)

⚙️ Data Preprocessing

Missing values removed from training data.

Dropped CustomerID as it’s not useful for prediction.

Encoding categorical columns using LabelEncoder:

Gender

Subscription Type

Contract Length

Scaling numerical features with StandardScaler.

SMOTE (Synthetic Minority Oversampling Technique) applied to handle class imbalance.

🤖 Machine Learning Models

The following models were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (best_k chosen)

Support Vector Machine

Gaussian Naive Bayes

Decision Tree

Random Forest

📊 Results
Performance on Validation (20% split from training data):
Algorithm	Train Accuracy	Test Accuracy
Decision Tree	1.0000	0.9962
Random Forest	1.0000	0.9962
Support Vector Machine	0.9705	0.9697
K-Nearest Neighbors	0.9697	0.9478
Gaussian Naive Bayes	0.9135	0.9058
Logistic Regression	0.8570	0.8519

✅ Random Forest & Decision Tree performed best on training/validation.
⚠️ Logistic Regression was chosen for deployment due to better generalization on unseen test data.

Performance on Final Test Data:
Algorithm	Test Data Accuracy
Logistic Regression	0.6007
Gaussian Naive Bayes	0.5810
K-Nearest Neighbors	0.5517
Support Vector Machine	0.5354
Random Forest	0.5078
Decision Tree	0.5077

🔹 Logistic Regression had the highest performance on unseen test dataset.

💾 Model Saving

The following objects were saved with joblib:

telco_churn_logreg_model.pkl → Trained Logistic Regression model

telco_churn_scaler.pkl → Scaler for numerical features

telco_churn_encoders.pkl → Label encoders for categorical features

🌐 Streamlit Web App

A user-friendly web application was built using Streamlit.

Features:

Interactive form for customer data entry

Automatic encoding & scaling of inputs

Model prediction with churn probability

Clear result display (stay vs churn)

📸 Visualizations

The following visualizations were created to better understand the dataset and model performance:

Churn Distribution Pie Chart → Shows the proportion of churned vs. non-churned customers.

Correlation Heatmap → Highlights relationships between numerical features and churn.

Histograms → Distribution of numerical features.

Boxplots → Compare numerical features against churn categories.

Countplots → Frequency distribution of categorical features (Gender, Subscription Type, Contract Length).

Accuracy Comparison Bar Chart → Compare model accuracies side by side.

👨‍💻 Author

Eslam Eid Abdallah
