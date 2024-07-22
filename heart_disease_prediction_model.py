import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

# Reading the dataset
df = pd.read_csv(r"C:\Users\nitin\heart_disease_prediction\data\heart.csv")

# Examining the data
print(df.head(10))
print(df.tail(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.columns)
print(df.isnull().any())

# Data Visualization
# Count plot
sns.countplot(x='target', data=df, palette=['blue', 'red'])
plt.show()

# Density plot
df.plot(kind='density', subplots=True, layout=(8, 2), sharex=False, figsize=(18, 20))
plt.show()

# Histogram
df.hist(figsize=(12, 10), color='turquoise')
plt.show()

# Box plot
df.plot(kind='box', subplots=True, layout=(7, 2), sharex=False, sharey=False, figsize=(18, 18), color='red')
plt.show()

# Correlation
correlation = df.corr()
print(correlation)

# KDE plot
sns.kdeplot(df['thalach'], shade=True, color='g')
plt.show()

# Heatmap
plt.figure(figsize=(18, 12))
plt.title('Correlation Heatmap plot')
sns.heatmap(correlation, square=True, annot=True)
plt.show()

# Distance plot
sns.displot(df['age'], bins=15, color='violet')
plt.show()

# Pair plot
feature = ['trestbps', 'age', 'chol', 'oldpeak', 'target', 'thalach']
sns.pairplot(df[feature], kind='scatter', diag_kind='hist')
plt.show()

# Distance plot
sns.displot(df['trestbps'], bins=10, kde=True, rug=True, color='g')
plt.show()

# Category plot
plt.figure(figsize=(8, 8))
sns.catplot(x='chol', kind='box', data=df, color='b')
plt.show()

# Kernel density estimation
plt.figure(figsize=(8, 8))
sns.kdeplot(data=df['thalach'], color='maroon')
plt.show()

# Feature Engineering and Feature Selection
categorical = []
continuous = []

for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical.append(column)
    else:
        continuous.append(column)

categorical.remove('target')
print(categorical)
print(continuous)

dataset = pd.get_dummies(df, columns=categorical)

cols = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
X = df[cols]
y = dataset['target']

# Modelling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
logreg = make_pipeline(StandardScaler(), LogisticRegression())
logreg.fit(X_train, y_train)
accuracy = accuracy_score(y_test, logreg.predict(X_test))
accuracy_rounded = round(accuracy * 100, 2)
print(f"Accuracy of Logistic regression is {accuracy_rounded}%")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
accuracy = accuracy_score(y_test, knn.predict(X_test))
accuracy_rounded = round(accuracy * 100, 2)
print(f"Accuracy of K-Nearest Neighbors is {accuracy_rounded}%")

# Finding the best k value
scoreList = []
for i in range(1, 20):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
plt.plot(range(1, 20), scoreList)
plt.xticks(np.arange(1, 20, 1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

# Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
accuracy = accuracy_score(y_test, svc.predict(X_test))
accuracy_rounded = round(accuracy * 100, 2)
print(f"Accuracy of Support Vector Machines(SVC) is {accuracy_rounded}%")

# Naive Bayes
GNB = GaussianNB()
GNB.fit(X_train, y_train)
accuracy = accuracy_score(y_test, GNB.predict(X_test))
accuracy_rounded = round(accuracy * 100, 2)
print(f"Accuracy of Naive Bayes is {accuracy_rounded}%")

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
accuracy = accuracy_score(y_test, decision_tree.predict(X_test))
accuracy_rounded = round(accuracy * 100, 2)
print(f"Accuracy of Decision Tree is {accuracy_rounded}%")

# Random Forest
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
model_accuracy = accuracy_score(y_test, classifier.predict(X_test))
accuracy_rounded = round(model_accuracy * 100, 2)
print(f"Accuracy of Random Forest is {accuracy_rounded}%")

# Comparing Models
accuracies = {
    'Logistic Regression': round(accuracy_score(y_test, logreg.predict(X_test)) * 100, 2),
    'K-Nearest Neighbors': round(accuracy_score(y_test, knn.predict(X_test)) * 100, 2),
    'Support Vector Machines(SVC)': round(accuracy_score(y_test, svc.predict(X_test)) * 100, 2),
    'Naive Bayes': round(accuracy_score(y_test, GNB.predict(X_test)) * 100, 2),
    'Decision Tree': round(accuracy_score(y_test, decision_tree.predict(X_test)) * 100, 2),
    'Random Forest': round(accuracy_score(y_test, classifier.predict(X_test)) * 100, 2)
}

print(accuracies)
colors = ["purple", "green", "orange", "magenta", "#CFC60E", "#0FBBAE"]
sns.set_style("whitegrid")
plt.figure(figsize=(18, 10))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel("Accuracy %", fontsize=20)
plt.xlabel("Algorithms", fontsize=20)
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

# Model Evaluation and Accuracy Measurement
y_head_lr = logreg.predict(X_test)
knn3 = KNeighborsClassifier(n_neighbors=10)
knn3.fit(X_train, y_train)
y_head_knn = knn3.predict(X_test)
y_head_svm = svc.predict(X_test)
y_head_nb = GNB.predict(X_test)
y_head_dtc = decision_tree.predict(X_test)
y_head_rf = classifier.predict(X_test)

# Measuring Accuracy using confusion matrix for all the algorithms
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_head_lr)
cm_knn = confusion_matrix(y_test, y_head_knn)
cm_svm = confusion_matrix(y_test, y_head_svm)
cm_nb = confusion_matrix(y_test, y_head_nb)
cm_dtc = confusion_matrix(y_test, y_head_dtc)
cm_rf = confusion_matrix(y_test, y_head_rf)

# Plotting confusion matrix for all the algorithms
plt.figure(figsize=(24, 12))
plt.suptitle("Confusion Matrices", fontsize=28)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.subplot(2, 3, 1)
plt.title("Logistic Regression Confusion Matrix", fontsize=20)
sns.heatmap(cm_lr, annot=True, cmap="Blues", fmt="d", cbar=True, annot_kws={"size": 24})

plt.subplot(2, 3, 2)
plt.title("K Nearest Neighbors Confusion Matrix", fontsize=20)
sns.heatmap(cm_knn, annot=True, cmap="BuPu", fmt="d", cbar=True, annot_kws={"size": 24})

plt.subplot(2, 3, 3)
plt.title("Support Vector Machines Confusion Matrix", fontsize=20)
sns.heatmap(cm_svm, annot=True, cmap="Greens", fmt="d", cbar=True, annot_kws={"size": 24})

plt.subplot(2, 3, 4)
plt.title("Naive Bayes' Confusion Matrix", fontsize=20)
sns.heatmap(cm_nb, annot=True, cmap="YlGnBu", fmt="d", cbar=True, annot_kws={"size": 24})

plt.subplot(2, 3, 5)
plt.title("Decision Tree Classifier Confusion Matrix", fontsize=20)
sns.heatmap(cm_dtc, annot=True, cmap="icefire", fmt="d", cbar=True, annot_kws={"size": 24})

plt.subplot(2, 3, 6)
plt.title("Random Forest Confusion Matrix", fontsize=20)
sns.heatmap(cm_rf, annot=True, cmap="flare", fmt="d", cbar=True, annot_kws={"size": 24})

plt.show()

# Exporting model using joblib library
joblib.dump(logreg, "hdp_model.pkl")

# Pipeline Creation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = {
    'numeric1': [1, 2, np.nan, 4],
    'numeric2': [10, 20, 30, np.nan],
    'category1': ['A', 'B', 'A', 'C'],
    'category2': ['X', 'Y', 'Y', 'X']
}
X = pd.DataFrame(data)

# Define columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Define transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline to the data
pipeline.fit(X)

# Accessing OneHotEncoder from the pipeline
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
print("Categorical Feature Names:", cat_feature_names)

# Load the saved model
loaded_model = joblib.load('hdp_model.pkl')

# Doublecheck model accuracy with classification report
y_pred = loaded_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

precision = (report['1']['precision'] + report['0']['precision']) / 2
recall = (report['1']['recall'] + report['0']['recall']) / 2
f1 = (report['1']['f1-score'] + report['0']['f1-score']) / 2

metrics_Random_Forest = {
    'Total Precision': precision,
    'Total Recall': recall,
    'Total F1-Score': f1,
    'Accuracy': report['accuracy']
}

metrics_df = pd.DataFrame(metrics_Random_Forest, index=['Random Forest Classifier'])
print(metrics_df)

# Saving models
destination = "toolkit"
if not os.path.exists(destination):
    os.makedirs(destination)

models = {"pipeline": pipeline}
for name, model in models.items():
    file_path = os.path.join(destination, f"{name}.joblib")
    joblib.dump(model, file_path)

# Load the pipeline from the .joblib file
pipeline = joblib.load('toolkit/pipeline.joblib')

# Use the pipeline to preprocess data or make predictions
