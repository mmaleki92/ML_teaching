from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# Load iris dataset and split it
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model to a file
dump(model, 'model.joblib')

# Later or in another script you can load the model
loaded_model = load('model.joblib')

# And use it to make predictions
predictions = loaded_model.predict(X_test)