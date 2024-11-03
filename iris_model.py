import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import pickle
import os

def train_and_save_model(csv_file):
    if model_exists():
        # print("A model already exists. Please delete 'iris_model.pkl' if you want to retrain the model.")
        return None  # Return None if model already exists

    try:
        # Load the dataset from CSV
        iris_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Separate features and target
    X = iris_df.iloc[:, :-1].values
    y = iris_df.iloc[:, -1].values

    # Create pie plot for flower distribution
    create_pie_plot(iris_df)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate accuracies
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Debug print
    # print(f"Trained model accuracy: {test_accuracy}")

    # Generate accuracy comparison chart
    create_accuracy_chart(model.score(X_train, y_train), test_accuracy)

    # Calculate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Save classification report as an image
    save_classification_report_as_image(class_report, y_test, y_pred)

    create_plots(y_test, y_pred, conf_matrix)

    # Save the model and its accuracy
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump((model, test_accuracy), f)

    return test_accuracy  # Return the accuracy for further use

def save_classification_report_as_image(class_report, y_test, y_pred):
    report_str = classification_report(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    plt.text(0.01, 1.25, report_str, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.title('Classification Report', fontsize=10)
    plt.savefig('static/classification_report.png', bbox_inches='tight', dpi=300)
    plt.close()


def create_pie_plot(iris_df):
    # Count the number of occurrences of each species
    flower_counts = iris_df.iloc[:, -1].value_counts()
    
    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(flower_counts, labels=flower_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Flower Species')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('static/flower_distribution.png')
    plt.close()

def create_accuracy_chart(train_accuracy, test_accuracy):
    # Create a bar chart for training vs testing accuracy
    plt.figure(figsize=(8, 6))
    labels = ['Training Accuracy', 'Testing Accuracy']
    accuracies = [train_accuracy, test_accuracy]
    
    plt.bar(labels, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 1)  # Set limit for y-axis to show accuracy as a percentage
    plt.ylabel('Accuracy')
    plt.title('Training vs Testing Accuracy')
    plt.grid(axis='y')
    plt.savefig('static/training_testing_accuracy.png')
    plt.close()

def create_plots(y_test, y_pred, conf_matrix):
    # Create the plot for True Labels vs Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Labels', marker='o', linestyle='-', color='blue', markersize=6)
    plt.plot(y_pred, label='Predictions', marker='x', linestyle='--', color='orange', markersize=8)
    plt.axhline(y=0.5, color='red', linestyle=':', label='Threshold')
    plt.title('True Labels vs Predictions', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Class Label', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('static/predictions_vs_true.png')
    plt.close()

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(rotation=45)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png')
    plt.close()

def model_exists():
    """ Check if a model has already been trained and saved. """
    return os.path.exists('iris_model.pkl')

if __name__ == "__main__":
    # Example usage, replace with actual path
    accuracy = train_and_save_model('path_to_your_csv.csv')
    # print(f"Model trained with accuracy: {accuracy}" if accuracy else "Model training failed.")
