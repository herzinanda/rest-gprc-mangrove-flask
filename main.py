from flask import Flask, request, send_file, jsonify
from flask_restful import Resource, Api
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Load your Gaussian process regression models
models = {
    'RBF': joblib.load('model_rbfKernel.pkl'),
    'Matern': joblib.load('model_maternKernel.pkl'),
    'RationalQuadratic': joblib.load('model_rationalQuadraticKernel.pkl')
}

# Create Flask app
app = Flask(__name__)
api = Api(app)

# Allow all origins for CORS
CORS(app, resources={r"/*": {"origins": "*"}})

class Prediction(Resource):
    def post(self):
        # Get JSON data from request
        json_data = request.get_json()
        
        # Get CSV URL and kernel from JSON data
        csv_url = json_data.get('csv_url')
        kernel = json_data.get('kernel')
        
        # Fetch CSV data from URL
        response = requests.get(csv_url)

        # Read CSV data into DataFrame
        csv_data = response.text
        df = pd.read_csv(StringIO(csv_data))

        print(df)
        
        # Define features and target
        features = ["ketinggianAir", "suhuAir", "suhuUdara", "kelembapanUdara", "tds", "orp", "do", "ph"]
        X_test = df[features].values
        target = "lokasi"
        y_test = df[target].values

        print(X_test, y_test)

        # Select the model based on the kernel
        model = models.get(kernel)
        if not model:
            return {'error': 'Invalid kernel selected'}, 400

        # Make prediction
        prediction = model.predict(X_test)
        print(prediction)
        report = classification_report(y_test, prediction, output_dict=True)
        print(report)

        conf_matrix = confusion_matrix(list(y_test), list(prediction))
        print(conf_matrix)

        # Store the confusion matrix and report in the app context
        request_conf_matrix = {
            'actual': y_test.tolist(),
            'predicted': prediction.tolist(),
            'report': report,
            'conf_matrix': conf_matrix.tolist()
        }
        app.config['last_conf_matrix'] = request_conf_matrix

        return jsonify(request_conf_matrix)

class ConfusionMatrixImage(Resource):
    def get(self):
        # Retrieve the confusion matrix from the app context
        conf_matrix = app.config.get('last_conf_matrix', {}).get('conf_matrix')

        if conf_matrix is None:
            return {'message': 'No confusion matrix available. Please make a prediction first.'}, 400

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Close the plot
        plt.close()

        return send_file(buf, mimetype='image/png', download_name='confusion_matrix.png')

# Add resources to API
api.add_resource(Prediction, '/predict')
api.add_resource(ConfusionMatrixImage, '/confusion_matrix_image')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
