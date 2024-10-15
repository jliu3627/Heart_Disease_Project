import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('cleaned_heart_2022.csv')

target_column = 'HadHeartAttack'
X = df.drop(columns=[target_column, 'Unnamed: 0'])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
class_weights = {0:1, 1:4}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=42, class_weight={0: class_weights[0], 1: class_weights[1]})
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
report = classification_report(y_test, y_pred)

print(report)