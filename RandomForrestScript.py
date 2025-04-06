#Jack's generic machine learning script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib

#Load data
# If using Downloads
# download_folder = str(os.path.join(Path.home(), "Downloads"))
# filepath = os.path.join(download_folder,fileName)

print("Starting Process")

filepath_1= 'C:/Users/bowes/Downloads/numerical_heart_data.csv'
filepath1= os.path.join(pathlib.Path(__file__).parent.resolve(), 'numerical_heart_data.csv')

filepath_2 = 'C:/Users/bowes/Downloads/numerical_stroke_data.csv'
filepath2 = os.path.join(pathlib.Path(__file__).parent.resolve(), 'numerical_stroke_data.csv')

print(f"Path:1 {filepath1}")
print(f"Path:2 {filepath2}")

#Prepare dataframe for numerical_heart_data
df = pd.read_csv(filepath1)
print(df.head())
x = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

#Prepare dataframe for numerical_stroke_data
#df = pd.read_csv(filepath2, skiprows=0, nrows=249*2)
#print(df.head())
#df = df.drop(columns=['id'])
#x = df.drop(columns=['stroke'])
#y = df['stroke']

#Train/test split [stratify adapted from scikit-learn.org]
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y if len(np.unique(y)) > 1 else None
)

#RandomForestClassifier [param_grid adapted from scikit-learn.org]
model = RandomForestClassifier(random_state=0)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

#Preprocessing with StandardScaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
])

#GridSearchCV (5-fold crossvalidation)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

#Best hyperparameters [best_model and hyperparameter printing adapted from scikit-learn.org]
print("\nBest Hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_    
y_pred = best_model.predict(X_test)

#Evaluateion
#Accuracy:
print("\nAccuracy: ")
print(accuracy_score(y_test, y_pred))

#Classification stats:
print("\nClassification stats: ")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()

print("TIME FOR ROC")
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# 4. Calculate AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {roc_auc:.2f}")

# 5. Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Chosen dataset')
plt.legend(loc="lower right")
plt.show()
