from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def load_decision_tree():
    try:
        with open('DecisionTree.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        raise RuntimeError("Failed to load Decision Tree model") from e

def load_svm():
    try:
        with open('svm_model.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        raise RuntimeError("Failed to load SVM model") from e

def load_rf():
    try:
        with open('RandomForest.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        raise RuntimeError("Failed to load Random Forest model") from e

def load_lr():
    try:
        with open('LogisticRegression.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        raise RuntimeError("Failed to load Logistic Regression model") from e

def load_nbc():
    try:
        with open('NBClassifier.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        raise RuntimeError("Failed to load Naive Bayes Classifier model") from e

models = {
    "decision_tree": load_decision_tree(),
    "svm": load_svm(),
    "random_forest": load_rf(),
    "naive_bayes": load_nbc(),
    "logistic_regression": load_lr(),
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'random_forest').lower()

        features = np.array([data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']])
        features = features.reshape(1, -1)

        predictions = {
            "decision_tree": None,
            "svm": None,
            "random_forest": None,
            "naive_bayes": None,
            "logistic_regression": None,
        }

        if model_name == "all":
            for name in models.keys():
                predictions[name] = models[name].predict(features)[0]
        elif model_name in models:
            predictions[model_name] = models[model_name].predict(features)[0]
        else:
            return jsonify({'error': 'Model not found'}), 400

        return jsonify({"decision_tree": predictions['decision_tree'],
    "logistic_regression": predictions['logistic_regression'],
    "naive_bayes": predictions['naive_bayes'],
    "random_forest": predictions['random_forest'],
    "svm": predictions['svm']})

    except KeyError as ke:
        return jsonify({'error': 'Missing feature in request', 'message': str(ke)}), 400
    except RuntimeError as re:
        return jsonify({'error': 'Model loading error', 'message': str(re)}), 500
    except Exception as e:
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
