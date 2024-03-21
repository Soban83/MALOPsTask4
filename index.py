from flask import Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load Iris dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv('iris.data', names=column_names)

# Split data into features and target variable
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Train classifier
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define a route to display the accuracy
@app.route('/accuracy')
def get_accuracy():
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)


 #http://127.0.0.1:5000/accuracy