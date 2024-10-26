import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from PIL import Image



# Custom CSS for background hover effect and button colors
st.markdown(""" 
    <style>
        body {
            background: linear-gradient(135deg, rgba(255,0,0,0.7), rgba(0,0,255,0.7));
            transition: background-color 0.5s ease;
        }
        .stButton > button:hover {
            background-color: rgba(255, 165, 0, 0.8);
            color: white;
        }
        .stButton > button {
            background-color: rgba(0, 255, 0, 0.7);
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 16px;
            transition: background-color 0.3s, transform 0.3s;
        }
        .stButton > button:active {
            transform: scale(0.95);
        }
        .creator-link {
            color: black;
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            display: block;  /* Make the link a block element */
            text-align: center;  /* Center text */
            margin-top: 10px;   /* Add some space above */
        }
        .creator-link:hover {
            background-color: white;  /* Background color on hover */
            color: red;              /* Change text color on hover */
            padding: 5px;           /* Add some padding on hover */
            border-radius: 5px;     /* Rounded corners */
        }
    </style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .download-button {
        display: inline-block;
        padding: 10px 20px;
        color: #fff;
        font-size: 16px;
        font-weight: bold;
        text-decoration: none;
        background: linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00ea);
        background-size: 600% 100%;
        border-radius: 8px;
        transition: background-position 1s ease;
        animation: animate 10s ease-in-out infinite;
    }
    
    .download-button:hover {
        background-position: 100% 0;
        box-shadow: 0px 4px 15px rgba(255, 255, 255, 0.5);
        text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.8);
    }
    
    @keyframes animate {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)



# App Title and Brief Description
st.sidebar.title("🎓 Campus Placement Prediction App")



# App description
st.sidebar.write("""
    Welcome to the **Campus Placement Prediction App**! 🎉

    This app leverages machine learning to predict a candidate's placement likelihood based on academic and skill attributes.
    A useful tool for students to assess their readiness and for recruiters to pre-screen potential candidates.
""")

st.sidebar.markdown(
        '<a href="https://www.linkedin.com/in/sm980/" class="creator-link">Created by SHASHWAT MISHRA</a>',
        unsafe_allow_html=True)

st.sidebar.write("### 🔍 How to Use This App:")
st.sidebar.write("""
1. **Enter Candidate Information**: Fill in your academic details, skills, and work experience in the Main sidebar. Include information like SSC and HSC board types and scores, degree type, CGPA, internships, projects, certifications, and more.

2. **Select a Model**: Choose from machine learning algorithms such as Random Forest, Logistic Regression, SVM, or Decision Tree to see how each performs on the dataset.

3. **Train the Model**: Click **"Train and Compare Models"** to initiate training. The app will preprocess data, balance classes using SMOTE, and display model accuracy, classification report, and feature importance analysis.

4. **Predict Placement**: Once the model is trained, enter your details and click **"Predict Placement"**. The app will estimate your likelihood of placement and provide a probability score for “Placed” or “Not Placed.”

""")




def train_placement_model(model_choice):
    st.write("🔍 **Training Placement Prediction Model...**")

    # Load the dataset
    data_path = 'Placement_Data.csv'
    df = pd.read_csv(data_path)

    # Display column names to debug missing columns
    st.write("Dataset Columns:", df.columns.tolist())  # Print available columns for debugging

    # Data Preprocessing
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})  # Encode gender
    df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})  # Encode placement status
    df['workex'] = df['workex'].map({'Yes': 1, 'No': 0})  # Encode work experience

    dummy_columns = ['degree_t', 'ssc_b', 'hsc_b', 'hsc_s']
    existing_columns = [col for col in dummy_columns if col in df.columns]
    
    if existing_columns:
        df = pd.get_dummies(df, columns=existing_columns, drop_first=True)
    else:
        st.write("⚠️ None of the dummy columns were found in the dataset.")


    # Define features (X) and target (y)
  # Define features (X) and target (y)
    X = df[['hsc_p', 'ssc_p', 'gender', 'workex', 'CGPA', 'Internships', 
            'Projects', 'Workshops/Certifications', 'AptitudeTestScore', 
            'SoftSkillsRating', 'AcademicConsistency', 'EngagementScore', 
            'PreparednessIndex', 'LeadershipScore']]
    y = df['status']


    # (Continue with the rest of the function as you have it)


    # Split the data into training and testing sets
    st.write("📊 **Splitting Data into Training and Testing Sets...**")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    st.write("⚖️ **Balancing Classes with SMOTE...**")
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Select the model based on user choice
    st.write(f"🔧 **Selected Model: {model_choice}**")
    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "SVM":
        model = SVC(probability=True)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()

    # Train the model
    st.write("🚀 **Training the Model...**")
    model.fit(X_resampled, y_resampled)
    st.success(f"✅ **Model trained and saved as 'placement_model.joblib' ({model_choice}).**")

    # Save the trained model to a file
    feature_names = X.columns.tolist()
    joblib.dump((model, feature_names), 'placement_model.joblib')

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"📈 **Model Accuracy: {accuracy * 100:.2f}%**")

    # Display classification report
    st.write("### 📋 Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

    # Plot feature importance
    plot_feature_importance(model, X, y)

    return model

# Function to plot feature importance
def plot_feature_importance(model, X, y):
    st.write("### 🔍 Feature Importance Analysis")
    with st.spinner("Calculating feature importance..."):
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance with Plotly
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig)

# Function to load the trained model
def load_model():
    try:
        model, feature_names = joblib.load('placement_model.joblib')
        return model, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model not found. Please train the model first.")
        return None, None


# Function for making placement predictions
def make_placement_prediction(model, feature_names, user_input):
    # Convert user_input to DataFrame and generate dummy variables
    user_input_df = pd.DataFrame([user_input])
    user_input_df = pd.get_dummies(user_input_df)
    
    # Align user_input_df columns with feature_names from the model
    user_input_df = user_input_df.reindex(columns=feature_names, fill_value=0)
    
    # Make predictions
    probabilities = model.predict_proba(user_input_df)[0]
    prediction = model.predict(user_input_df)
    return prediction, probabilities

# Streamlit app interface
st.header("📝 Input Features")
st.write("📌 Please enter your academic and skill-related information in the fields below.")

# Model choice selection
st.subheader("🛠️ Model Selection")
st.write("Choose a machine learning model for training:")
model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"])

# Input fields for user to enter values
st.subheader("🎓 Enter Your Academic Details")
st.write("Provide your academic scores and details:")

ssc_b = st.selectbox("📘 SSC Board", options=["Central", "Others"])
ssc_p = st.slider("📈 SSC Percentage (%)", 0, 100, 50)

hsc_s = st.selectbox("📚 HSC Specialization", options=["Commerce", "Science", "Arts"])
hsc_b = st.selectbox("📙 HSC Board", options=["Central", "Others"])
hsc_p = st.slider("📊 HSC Percentage (%)", 0, 100, 50)

gender = st.selectbox("👤 Gender", options=["Male", "Female"])
work_experience = st.selectbox("💼 Work Experience", options=["Yes", "No"])

# Additional inputs
st.subheader("🛠️ Skills and Experiences")
st.write("Provide details about your skills and relevant experiences:")


degree_t = st.selectbox("🎓 Degree Type", options=["Sci&Tech", "Comm&Mgmt", "Others"])
CGPA = st.slider("🎓 CGPA", 0.0, 10.0, 7.5)
internships = st.number_input("💼 Number of Internships", min_value=0, value=0)
projects = st.number_input("🔬 Number of Projects", min_value=0, value=0)
workshops_certifications = st.number_input("📜 Number of Workshops/Certifications", min_value=0, value=0)
aptitude_test_score = st.slider("🧠 Aptitude Test Score", 0, 100, 50)
soft_skills_rating = st.slider("🤝 Soft Skills Rating (0-5)", 0, 5, 3)
academic_consistency = st.slider("📐 Academic Consistency (-20 to 20)", -20.0, 20.0, 0.0)
engagement_score = st.slider("📊 Engagement Score (0-10)", 0, 10, 5)
preparedness_index = st.slider("📝 Preparedness Index (0-100)", 0, 100, 50)
leadership_score = st.number_input("🌟 Leadership Score (0-5)", min_value=0, max_value=5, value=0)

# Map inputs
board_map = {'Central': 0, 'Others': 1}
specialization_map = {'Commerce': 0, 'Science': 1, 'Arts': 2}
degree_type_map = {'Sci&Tech': 0, 'Comm&Mgmt': 1, 'Others': 2}
gender_map = {'Male': 0, 'Female': 1}
workex_map = {'Yes': 1, 'No': 0}

# Prepare user input as a list for placement prediction
user_input = [
    hsc_p,
    ssc_p,
    gender_map[gender],
    board_map[ssc_b],
    board_map[hsc_b],
    specialization_map[hsc_s],
    degree_type_map[degree_t],
    workex_map[work_experience],
    CGPA,
    internships,
    projects,
    workshops_certifications,
    aptitude_test_score,
    soft_skills_rating,
    academic_consistency,
    engagement_score,
    preparedness_index,
    leadership_score
]

# Train model and predict placement
st.write("## 🚀 Train Model and Predict Placement")
st.write("Click the button below to train the model based on your selected parameters and predict placement status.")

if st.button("🔄 Train and Compare Models"):
    model = train_placement_model(model_choice)

st.write("After training, click the button below to predict placement status based on the entered details.")
if st.button("🔮 Predict Placement"):
    model, feature_names = load_model()
    if model:
        prediction, probabilities = make_placement_prediction(model, feature_names, user_input)
        status = "Placed" if prediction[0] == 1 else "Not Placed"
        probability = probabilities[1] if prediction[0] == 1 else probabilities[0]
        st.write(f"### 📊 Prediction Result: {status} (Probability: {probability * 100:.2f}%)")
