#Jack's generic machine learning script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
filepath = 'C:/Users/bowes/Downloads/combined_diabetes_dataset.xlsx'
df = pd.DataFrame()
if filepath.endswith(('.xls', '.xlsx')):
    df = pd.read_excel(filepath)
print(df.head())

# Prepare dataframes
x = df.drop(columns=['Outcome'])
y = df['Outcome']  # Dependent variable

# Train and test sets [stratify adapted from scikit-learn.org]
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y if len(np.unique(y)) > 1 else None
)

# RandomForestClassifier [param_grid adapted from scikit-learn.org]
model = RandomForestClassifier(random_state=0)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Preprocessing with StandardScaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
])

# GridSearchCV (5-fold crossvalidation)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best hyperparameters [best_model and hyperparameter printing adapted from scikit-learn.org]
print("\nBest Hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_    
y_pred = best_model.predict(X_test)

# Evaluateion
# Accuracy:
print("\nAccuracy: ")
print(accuracy_score(y_test, y_pred))
# Classification stats:
print("\nClassification stats: ")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()
plt.show()