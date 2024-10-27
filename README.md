# 🎓 Campus Placement Prediction App

The Campus Placement Prediction App is an interactive Streamlit application designed to predict the likelihood of a candidate's placement based on academic, experiential, and skill-based inputs. This app is beneficial for students assessing placement readiness and for recruiters pre-screening candidates. It provides a visually engaging interface, enhanced with custom CSS styling and animations.

# Table of Contents

Features
Installation
Usage
How It Works
Technologies Used
License

# Features

Interactive Model Training: Select and train various machine learning models, including Random Forest, Logistic Regression, SVM, and Decision Tree.

Predictive Analysis: Enter candidate details and predict placement likelihood with model probabilities.

Custom Styling: Enhanced UI/UX with CSS for gradient backgrounds, hover effects, and animation.

Data Insights: Display model accuracy, classification report, and feature importance for transparency and better decision-making.

# Installation

Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/campus-placement-prediction.git
cd campus-placement-prediction
Install Requirements: Make sure you have Python installed, then install the dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Application:

bash
Copy code
streamlit run app.py

# Usage

Launch the Application: Start the app by running streamlit run app.py.

Navigate Through the Sidebar:

Enter Candidate Information: Fill in academic, skill, and experience fields.

Model Selection: Choose the desired machine learning model.

Train the Model: Click "Train and Compare Models" to analyze and train the model.

Predict Placement: Use the entered candidate details to predict placement status.

# How It Works

Data Loading and Preprocessing:

Loads data from the Placement_Data.csv file.

Encodes categorical variables such as gender, work experience, and degree types.

Balances the data using SMOTE to handle class imbalances.

Model Training and Comparison:

Provides a selection of algorithms (Random Forest, Logistic Regression, SVM, Decision Tree).

Trains and saves the selected model (placement_model.joblib), displaying accuracy and classification metrics.

# Prediction:

Users can enter new candidate data, and the app calculates the probability and placement likelihood.

Visualizes feature importance to give insight into the most influential factors for placement.

# Technologies Used: 

Python Libraries: pandas, scikit-learn, joblib, plotly, imblearn, Pillow

Streamlit: For the interactive front end and visualization.

Plotly: For dynamic data visualization.

CSS: Custom styling for enhanced user experience.

# License
This project is licensed under the Apace-2.0 License.
