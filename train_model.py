
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

df = pd.read_csv('data/credit_data.csv')
df.fillna(method='ffill', inplace=True)
df['employment_status'] = LabelEncoder().fit_transform(df['employment_status'])
df['debt_to_income'] = df['debt'] / (df['income'] + 1)

X = df.drop('creditworthy', axis=1)
y = df['creditworthy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'models/RandomForest.pkl')
