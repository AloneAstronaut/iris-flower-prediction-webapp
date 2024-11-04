Iris Flower Prediction Web App
Overview

This web application predicts the species of Iris flowers based on user-input features using a K-Nearest Neighbors (KNN) classifier. It provides user authentication, model training, and the ability to visualize model performance metrics.
Features

    User registration and login functionality.
    Upload CSV datasets for model training.
    Predict flower species based on user input.
    Display model accuracy and various visualizations:
        Pie chart of flower species distribution.
        Bar chart comparing training and testing accuracy.
        Line plot of true labels vs predictions.
        Confusion matrix heatmap.
    Session management for tracking logged-in users and model status.

Technologies Used

    Backend:
        Flask: Web framework for Python.
        SQLite: Database for storing user credentials and session data.
        scikit-learn: For model training and prediction.
        Matplotlib: For data visualization.
        Pandas and NumPy: For data manipulation.
    Frontend:
        HTML: For structuring web pages.
        CSS: For styling web pages.
        JavaScript: For adding interactivity to web pages.

K-Nearest Neighbors (KNN) Classifier

The KNN classifier is a simple yet effective supervised learning algorithm used for classification tasks. It operates on the principle of finding the ‘k’ closest data points in the feature space and predicting the class of the input data based on the majority class among those neighbors.
How KNN Works:

    Distance Calculation: For a given test instance, the distance to all training samples is calculated (commonly using Euclidean distance).
    Finding Neighbors: The k training samples closest to the test instance are identified.
    Voting: The most frequent class label among these k neighbors is chosen as the predicted class for the test instance.

References to Learn More:

    Introduction to KNN
    KNN Classifier Documentation (scikit-learn)
    Understanding KNN in Machine Learning

Requirements

Ensure you have Python 3.x installed, along with the necessary libraries:

bash

pip install Flask pandas numpy matplotlib scikit-learn

Code Structure

arduino

📁 Dataset
    📄 dataset.py
    📄 iris_dataset.csv
📄 app.py
📄 database.py
📄 iris_model.pkl
📄 iris_model.py
📄 requirements.txt
📁 static
    📄 classification_report.png
    📄 confusion_matrix.png
    📄 flower_distribution.png
    📄 predictions_vs_true.png
    📄 training_testing_accuracy.png
📁 templates
    📄 charts.html
    📄 index.html
    📄 login.html
    📄 main.html
    📄 predict.html
    📄 remote_users.html
    📄 result.html
    📄 upload.html
📁 uploads
    📄 iris_dataset.csv
📄 users.db

    Dataset: Contains the dataset and any related scripts.
    static: Stores generated visualizations.
    templates: Contains HTML templates for the web app.
    app.py: The main Flask application.
    database.py: Database handling functions.
    iris_model.py: Functions for training and saving the KNN model.
    requirements.txt: List of dependencies.

Setup

    Clone the Repository: Start by cloning the repository to your local machine:

    bash

git clone https://github.com/your-username/iris-flower-prediction-webapp.git
cd iris-flower-prediction-webapp

Install Required Libraries: Make sure you have all the required Python libraries installed by running:

bash

pip install -r requirements.txt

Prepare Your Dataset: You can use the provided dataset or download it from the following link: iris_dataset.csv. Ensure the dataset is in CSV format with features including sepal length, sepal width, petal length, and petal width.

Run the Application: Start the Flask application by running:

bash

    python app.py

    Access the App: Open your web browser and navigate to http://127.0.0.1:5000/ to access the application.

Endpoints

    /: Main landing page.
    /index: Displays the model training status and accuracy.
    /login: User login page.
    /register: User registration endpoint (POST).
    /upload: Page for uploading datasets for model training.
    /predict: Page for making predictions based on user input.
    /charts: Displays accuracy metrics and visualizations.
    /remote_users: Lists currently logged-in users from remote addresses.
    /logout: Logs out the current user.

Usage

    Register: Create an account via the login page.
    Log In: Enter your credentials to access the app.
    Upload Dataset: Navigate to the upload page to train the model with your dataset.
    Make Predictions: Use the prediction page to input feature values and obtain species predictions.

Visualizations

The application generates the following visualizations, which are saved in the static folder:

    flower_distribution.png: Pie chart showing the distribution of flower species.
    training_testing_accuracy.png: Bar chart comparing training and testing accuracy.
    predictions_vs_true.png: Line plot of true labels vs predictions.
    confusion_matrix.png: Heatmap of the confusion matrix.
    classification_report.png: Visual representation of the classification report.

License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

    Flask
    Scikit-learn
    Matplotlib
    Pandas
    SQLite
