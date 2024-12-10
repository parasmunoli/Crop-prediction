from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from PIL import Image
import joblib
import numpy as np
import tensorflow as tf
from collections import Counter

app = Flask(__name__)
CORS(app)

# Load machine learning models for crop prediction
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
def crop_predict():
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
            prediction_counts = Counter(predictions.values())
            most_common_prediction, frequency = prediction_counts.most_common(1)[0]
            prediction = most_common_prediction
            return jsonify({"All": prediction})
        elif model_name in crop_models:
            predictions[model_name] = crop_models[model_name].predict(features)[0]
        else:
            return jsonify({'error': 'Model not found'}), 400

        return jsonify(predictions)

    except KeyError as ke:
        return jsonify({'error': 'Missing feature in request', 'message': str(ke)}), 400
    except RuntimeError as re:
        return jsonify({'error': 'Model loading error', 'message': str(re)}), 500
    except Exception as e:
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

model_paths = {
    'inception': 'inception_model.tflite',
    'densenet': 'densenet_model.tflite',
    'vgg16': 'VGG16_model.tflite'
}

models = {}
for name, model_path in model_paths.items():
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        models[name] = interpreter
    except Exception as e:
        print(f"Error loading model {name}: {e}")

class_labels = [
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Healthy",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic virus",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus"
]

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

def get_model_prediction(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(prediction, axis=1)[0]

@app.route('/tomato_predict', methods=['POST'])
def image_predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected image file'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if image_file.filename.split('.')[-1].lower() in allowed_extensions:
            image = Image.open(image_file)
            preprocessed_image = preprocess_image(image, target_size=(224, 224))

            model_choice = request.form.get('model', 'all')
            predictions = {}

            if model_choice == 'all':
                for name, interpreter in models.items():
                    try:
                        predictions[name] = class_labels[get_model_prediction(interpreter, preprocessed_image)]
                    except Exception as e:
                        predictions[name] = f"Error: {str(e)}"
                return jsonify(predictions)

            try:
                selected_model = models[model_choice]
                prediction = class_labels[get_model_prediction(selected_model, preprocessed_image)]
                return jsonify({f'{model_choice}_prediction': prediction})
            except KeyError:
                return jsonify({'error': f'Invalid model choice. Available options are: {", ".join(models.keys())}, or "all".'}), 400
            except Exception as e:
                return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        return jsonify({'error': 'Invalid image file format'}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"success": True, "code": 200})

if __name__ == '__main__':
    app.run(debug=True)
