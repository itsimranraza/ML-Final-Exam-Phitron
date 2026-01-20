import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv("diabetes.csv")
print(df.head())
df.shape

#1
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols]=df[cols].replace(0,np.nan)

#2
df[cols]=df[cols].fillna(df[cols].median())

#3
X=df.drop("Outcome",axis=1)
y=df["Outcome"]

#4
Q1=X.quantile(0.25)
Q3=X.quantile(0.75)
IQR=Q3-Q1

upper_bound=Q3+1.5*IQR
lower_bound=Q1-1.5*IQR

is_outlier=(X<lower_bound) | (X>upper_bound)

outlier_rows=is_outlier.any(axis=1)

mask=(outlier_rows==False)
X=X[mask]
y=y[mask]

#5
X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("clf", LogisticRegression(max_iter=500))
])


pipeline.fit(X_train, y_train)

cv_scores=cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

cv_scores.mean(),cv_scores.std()

param_grid={
    "clf__C":[0.01,0.1,1,10],
    "clf__solver":["liblinear", "lbfgs"]
}

grid=GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
grid.best_params_,grid.best_score_

best_model=grid.best_estimator_

y_pred=best_model.predict(X_test)

print("Accuracy",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

with open("diabetes_pipeline.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("pipeline saved as diabetes_pipeline.pkl")
