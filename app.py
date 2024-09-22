from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models with try-except to handle loading issues
def load_decision_tree():
    try:
        with open('DecisionTree.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        return str(e)

def load_svm():
    try:
        with open('svm_model.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        return str(e)

def load_rf():
    try:
        with open('RandomForest.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        return str(e)
    
def load_lr():
    try:
        with open('LogisticRegression.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        return str(e)

def load_nbc():
    try:
        with open('NBClassifier.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        return str(e)

# Load models
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
        # Parse the incoming JSON data
        data = request.get_json(force=True)

        # Extract the model name and features
        model_name = data.get('model', 'random_forest').lower()
        features = np.array([data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']])
        features = features.reshape(1, -1)
        
        predictions = {}

        # If "all" models requested, predict using all models
        if model_name == "all":
            for name, model in models.items():
                if isinstance(model, str):  # Check if the model failed to load
                    predictions[name] = f"Error: {model}"
                else:
                    predictions[name] = model.predict(features)[0]
        else:
            if model_name not in models:
                return jsonify({'error': 'Model not found'}), 400
            
            model = models[model_name]
            if isinstance(model, str):  # Check if the model failed to load
                return jsonify({'error': f"Error loading {model_name} model: {model}"}), 500
            predictions[model_name] = model.predict(features)[0]
    
        return jsonify(predictions)
    
    except Exception as e:
        # If any error occurs, return an internal server error message
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    try:
        return jsonify({'success': True, 'code': 200}), 200
    except Exception as e:
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

# Global error handler for internal server errors
@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
