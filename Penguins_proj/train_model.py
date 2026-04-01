import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("penguins.csv").dropna()

le = preprocessing.LabelEncoder()
X = data[["bill_length_mm", "flipper_length_mm"]]
y = le.fit_transform(data["species"])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=11))
])

clf.fit(X_train, y_train)

joblib.dump(clf, "model.joblib")
joblib.dump(le, "label_encoder.joblib")
