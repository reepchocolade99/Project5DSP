from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no image provided'}), 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    # Placeholder: load your model and run inference on `path`
    # For now render a result page with stakeholder text describing a truck being towed case.
    result_text = (
        "Stakeholder report: A medium-duty truck was found immobilized at the loading bay after mechanical failure. "
        "The vehicle was obstructing access and required a tow to a secure facility. Towing was performed by a licensed operator; "
        "the vehicle was secured and transported to the impound area for inspection. No hazardous materials were observed at the scene."
    )
    return render_template('result.html', text=result_text, filename=filename)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
