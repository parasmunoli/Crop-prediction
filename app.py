from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from PIL import Image
import joblib
import gdown
import numpy as np

app = Flask(__name__)
CORS(app)

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

crop_models = {
    "decision_tree": load_decision_tree(),
    "svm": load_svm(),
    "random_forest": load_rf(),
    "naive_bayes": load_nbc(),
    "logistic_regression": load_lr(),
}

@app.route('/crop_predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'random_forest').lower()

        features = np.array([
    round(float(data['N']), 2),
    round(float(data['P']), 2),
    round(float(data['K']), 2),
    round(float(data['temperature']), 2),
    round(float(data['humidity']), 2),
    round(float(data['ph']), 2),
    round(float(data['rainfall']), 2)
])

        features = features.reshape(1, -1)

        predictions = {
            "decision_tree": None,
            "svm": None,
            "random_forest": None,
            "naive_bayes": None,
            "logistic_regression": None,
        }

        if model_name == "all":
            for name in crop_models.keys():
                predictions[name] = crop_models[name].predict(features)[0]
        elif model_name in crop_models:
            predictions[model_name] = crop_models[model_name].predict(features)[0]
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

# model_ids = {
#     'densenet': '1OYIlqOvuQgD4OLKA0HAa5HCb_GovcWvI',
#     'inception': '1JQQkD-8jRQXI9Twp7BjJu_kxlmOqiljy',
#     'vgg16': '1bHoCvQIs7_jI12W50CyREDZHKnZwavnt'
# }

# class_labels = [
#     "Tomato Bacterial spot",
#     "Tomato Early blight",
#     "Tomato Healthy",
#     "Tomato Late blight",
#     "Tomato Leaf Mold",
#     "Tomato Mosaic virus",
#     "Tomato Septoria leaf spot",
#     "Tomato Spider mites",
#     "Tomato Target Spot",
#     "Tomato Yellow Leaf Curl Virus"
# ]

# def load_model_from_drive(file_id):
#     url = f'https://drive.google.com/uc?id={file_id}'
#     model_data = gdown.download(url, None, quiet=True)
#     model = joblib.load(model_data)
#     return model

# models = {name: load_model_from_drive(file_id) for name, file_id in model_ids.items()}

# def preprocess_image(image, target_size):
#     image = image.resize(target_size)
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# def get_model_prediction(model, image):
#     prediction = model.predict(image)
#     return np.argmax(prediction, axis=1)[0]

# def make_prediction(modelname, image):
#     prediction_index = get_model_prediction(models[modelname], image)
#     return class_labels[prediction_index]

# @app.route('/tomato_predict', methods=['POST'])
# def predict_single_model():
#     modelname = request.form.get('modelname')
#     if not modelname or modelname not in models:
#         return jsonify({'error': 'Invalid or missing model name provided'}), 400

#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     image_file = request.files['image']
#     try:
#         image = Image.open(image_file)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

#     preprocessed_image = preprocess_image(image, target_size=(224, 224))
#     prediction = make_prediction(modelname, preprocessed_image)
#     return jsonify({'model': modelname, 'prediction': prediction})

# @app.route('/tomato_predict/all', methods=['POST'])
# def predict_all():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400

#     image_file = request.files['image']
#     try:
#         image = Image.open(image_file)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

#     preprocessed_image = preprocess_image(image, target_size=(224, 224))
#     predictions = {}
#     for modelname in models.keys():
#         predictions[modelname] = make_prediction(modelname, preprocessed_image)

#     return jsonify(predictions)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"success": True, "code": 200})

if __name__ == '__main__':
    app.run(debug=True)
