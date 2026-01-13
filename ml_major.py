# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, plot_confusion_matrix

# Load Dataset
data = pd.read_csv('lending_club_loan_dataset.csv', low_memory=False)

# Exploratory Data Analysis
print(data.describe())
print(data.info())

# Handling Missing Values
data['annual_inc'].fillna(data['annual_inc'].mean(), inplace=True)
data['home_ownership'].fillna(data['home_ownership'].mode()[0], inplace=True)
data['dti'].fillna(data['dti'].mean(), inplace=True)
data.drop('last_major_derog_none', axis=1, inplace=True)

# Removing Outliers
Q1 = data['annual_inc'].quantile(0.25)
Q3 = data['annual_inc'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['annual_inc'] < (Q1 - 1.5 * IQR)) | (data['annual_inc'] > (Q3 + 1.5 * IQR)))]

# Feature Engineering
data['grade'] = data['grade'].map({"A":7, "B":6, "C":5, "D":4, "E":3, "F":2, "G":1})
data = pd.get_dummies(data, columns=['term', 'home_ownership', 'purpose'], drop_first=True)

# Split Dataset
X = data.drop(['bad_loan'], axis=1)
y = data['bad_loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f"Logistic Regression AUC: {auc_lr}")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=47)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict_proba(X_test)[:, 1]
auc_knn = roc_auc_score(y_test, y_pred_knn)
print(f"KNN AUC: {auc_knn}")

# Support Vector Machine
svc = SVC(probability=True, class_weight='balanced')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict_proba(X_test)[:, 1]
auc_svc = roc_auc_score(y_test, y_pred_svc)
print(f"SVC AUC: {auc_svc}")

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest AUC: {auc_rf}")

# Plot ROC Curve
def plot_roc(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc_score(y_true, y_scores):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

plot_roc(y_test, y_pred_lr, "Logistic Regression")
plot_roc(y_test, y_pred_knn, "K-Nearest Neighbors")
plot_roc(y_test, y_pred_svc, "Support Vector Machine")
plot_roc(y_test, y_pred_rf, "Random Forest")
