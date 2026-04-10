from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# --------------------------
# Load trained SVM model
# --------------------------
with open('models/svm_model.pkl', 'rb') as f:
    svm_data = pickle.load(f)

svm_model = svm_data['model']
scaler = svm_data['scaler']
CLASSES = svm_data['classes']

# --------------------------
# Dummy users database
# --------------------------
users = {}  # {email: password}

# --------------------------
# Feature extraction function
# --------------------------
def extract_features(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (128, 128))
    img = cv2.equalizeHist(img)

    img_scaled = (img / 8).astype(np.uint8)
    glcm = graycomatrix(img_scaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.extend(graycoprops(glcm, prop).flatten())

    moments = cv2.HuMoments(cv2.moments(img)).flatten()
    features.extend(moments)

    img_color = cv2.imread(image_path)
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, (32, 32))
    features.extend(np.histogram(hsv[:, :, 0], bins=32, density=True)[0])

    return np.array(features)

# --------------------------
# Routes
# --------------------------

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email] == password:
            session['email'] = email
            return redirect(url_for('dashboard'))
        else:
            return render_template('index.html', error="Invalid credentials")
    return render_template('index.html', error=None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users:
            return render_template('register.html', error="Email already exists")
        users[email] = password
        return redirect(url_for('login'))
    return render_template('register.html', error=None)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    prediction = None
    confidence = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        file_path = "temp.jpg"
        file.save(file_path)

        try:
            features = extract_features(file_path)
            features_scaled = scaler.transform([features])
            probs = svm_model.predict_proba(features_scaled)[0]
            idx = int(np.argmax(probs))
            prediction = CLASSES[idx]
            confidence = round(float(np.max(probs)) * 100, 2)
        except Exception as e:
            prediction = str(e)
            confidence = 0
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return render_template('dashboard.html', prediction=prediction, confidence=confidence)

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

# --------------------------
# Run Flask app
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)