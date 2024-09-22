from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

def load_decision_tree():
    with open('DecisionTree.pkl', 'rb') as model_file:
        return pickle.load(model_file)

def load_svm():
    with open('svm_model.pkl', 'rb') as model_file:
        return pickle.load(model_file)

def load_rf():
    with open('RandomForest.pkl', 'rb') as model_file:
        return pickle.load(model_file)
    
def load_lr():
    with open('LogisticRegression.pkl', 'rb') as model_file:
        return pickle.load(model_file)

def load_nbc():
    with open('NBClassifier.pkl', 'rb') as model_file:
        return pickle.load(model_file)

models = {
    "decision_tree": load_decision_tree(),
    "svm": load_svm(),
    "random_forest": load_rf(),
    "naive_bayes": load_nbc(),
    "logistic_regression": load_lr(),
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    model_name = data.get('model', 'random forest').lower()

    features = np.array([data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']])
    features = features.reshape(1, -1)
    
    predictions = {}

    if model_name == "all":
        for name, model in models.items():
            predictions[name] = model.predict(features)[0]
    else:
        if model_name not in models:
            return jsonify({'error': 'Model not found'}), 400
        
        model = models[model_name]
        predictions[model_name] = model.predict(features)[0] 
 
    return jsonify({predictions})
    
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'success': True, 'code': 200}), 200

if __name__ == '__main__':
    app.run(debug=True)
