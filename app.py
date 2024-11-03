from flask import Flask, request, render_template, jsonify, url_for, redirect, session
import pickle
import numpy as np
import os
from database import init_db, add_user, validate_user, add_remote_user, fetch_remote_users
from iris_model import train_and_save_model, model_exists
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = 'Jg!^LzB`c$79=vC4s#5d6ar./'  # Set your secret key here

# Initialize the database here
init_db()

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index')
def index():
    model_trained = session.get('model_trained', False)
    accuracy = session.get('accuracy', None)
    return render_template('index.html', model_trained=model_trained, accuracy=accuracy)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if validate_user(username, password):
            add_remote_user(username, request.remote_addr)
            session['username'] = username
            session['remote_users'] = fetch_remote_users()  # Store remote users in session
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['new_username']
    password = request.form['new_password']
    if add_user(username, password):
        return redirect(url_for('login'))
    else:
        return 'Username already exists', 400

@app.route('/logout')
def logout():
    # Clear all relevant session data
    session.pop('username', None)
    session.pop('remote_users', None)  # Clear remote users on logout
    session.pop('model_trained', None)  # Clear model trained status on logout
    session.pop('accuracy', None)  # Clear accuracy on logout

    # Optionally, clear all session data (if you want to ensure a full reset)
    session.clear()

    return redirect(url_for('main'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            if model_exists():  # Check if the model exists
                # print("Loading existing model...")
                model, accuracy = load_model()  # Load the model and accuracy
                print(accuracy)

                # Update session
                session['model_trained'] = True
                session['accuracy'] = int(accuracy * 100)
                # print(f"Loaded model accuracy: {accuracy}")

                return jsonify({"success": True, "message": "Model trained successfully.", "accuracy": accuracy})

            # Train the model if it doesn't exist
            accuracy = train_and_save_model(file_path)  # Capture the returned accuracy

            # Update training status in session
            session['model_trained'] = True
            session['accuracy'] = int(accuracy * 100)
            # print(f"Trained model accuracy: {accuracy}")

            return jsonify({"success": True, "message": "Model trained successfully."})
    
    return render_template('upload.html')


@app.route('/clear_model_status', methods=['POST'])
def clear_model_status():
    session['model_trained'] = False
    session['accuracy'] = None  # Reset training status
    return jsonify(success=True)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model, accuracy = load_model()  # Unpack model and accuracy
    if model is None:
        return "Model not trained yet!", 400
    
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)

        species = ['Setosa', 'Versicolor', 'Virginica']
        return render_template('result.html', prediction=species[prediction[0]])
    
    return render_template('predict.html')  # Render the prediction form if GET request


@app.route('/charts', methods=['GET'])
def get_charts():
    accuracy_percentage = session.get('accuracy', "Not available")
    return render_template('charts.html', accuracy=accuracy_percentage)

@app.route('/remote_users', methods=['GET'])
def remote_users():
    remote_users = fetch_remote_users()
    return render_template('remote_users.html', remote_users=remote_users)

def load_model():
    if os.path.exists('iris_model.pkl'):
        with open('iris_model.pkl', 'rb') as f:
            model, accuracy = pickle.load(f)  # Unpack the model and accuracy
            return model, accuracy
    return None, None  # Return None for both if model doesn't exist


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Use the port specified by Render
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug to False for production
