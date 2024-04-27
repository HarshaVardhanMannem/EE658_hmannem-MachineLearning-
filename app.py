# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PIL import Image

# Load IRIS and Digits datasets from sklearn.datasets
def load_datasets(selected_dataset):
    if selected_dataset == "IRIS":
        return datasets.load_iris()
    elif selected_dataset == "Digits":
        return datasets.load_digits()
    else:
        raise ValueError("Invalid dataset selected")

# Train the classifier
def train_classifier(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

# Make predictions
def predict(classifier, input_data):
    return classifier.predict(input_data)

# User Interface
def main():
    st.title("Machine Learning Model Prediction")
    # Set up sidebar for dataset and model selection
    selected_dataset = st.sidebar.radio("Select Dataset", ("IRIS", "Digits"))
    selected_classifier = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "Neural Networks", "Naive Bayes"))

    data = load_datasets(selected_dataset)
    X = data.data
    y = data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load selected classifier
    if selected_classifier == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000)  # Increase max_iter
    elif selected_classifier == "Neural Networks":
        classifier = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu', solver='adam', max_iter=500)
        # Adjust parameters for MLPClassifier
    elif selected_classifier == "Naive Bayes":
        classifier = GaussianNB()

    # Scale the data if using Logistic Regression or Neural Networks
    if selected_classifier in ["Logistic Regression", "Neural Networks"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # 4. User Input
    if selected_dataset == "Digits":
        # Take input as image
        st.title("Upload an Image of a Digit")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Convert image to grayscale and resize to 8x8 (compatible with Digits dataset)
            image = image.convert('L').resize((8, 8))
            image_array = np.array(image)

            # Flatten image array to 1D and normalize pixel values
            flattened_image = image_array.flatten() / 255.0

            # Display flattened image
            #st.image(image_array, caption='Flattened Image.', use_column_width=True)

            # Make predictions
            if st.button("Predict"):
                input_data = np.array([flattened_image])

                # Train the classifier
                trained_classifier = train_classifier(classifier, X_train_scaled, y_train)

                # Make predictions
                prediction = predict(trained_classifier, input_data)

                # Display prediction result
                st.title("Prediction Result")
                st.write(f"The predicted digit is: {prediction[0]}")

                # Evaluate model accuracy on test data
                y_pred = trained_classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {accuracy:.2f}")
    else:
        # Dynamically generate input fields for feature values
        st.title("Feature Inputs")
        user_inputs = {}
        for i, feature_name in enumerate(data.feature_names):
            user_input = st.number_input(feature_name, value=0.0)
            user_inputs[feature_name] = user_input

        if st.button("Predict"):
            input_data = pd.DataFrame([user_inputs], columns=data.feature_names)

            # Train the classifier
            trained_classifier = train_classifier(classifier, X_train_scaled, y_train)

            # Make predictions
            prediction = predict(trained_classifier, input_data)

            # Display prediction result
            st.title("Prediction Result")
            st.write(f"The predicted class is: {prediction[0]}")

            # Evaluate model accuracy on test data
            y_pred = trained_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
