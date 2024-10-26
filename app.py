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

# Custom CSS for hover effects on buttons
st.markdown("""
    <style>
    .stButton>button:hover {
        background-color: rgb(255, 255, 255);
        box-shadow: 0 0 10px rgba(255,255,255, 0.5), 
                    0 0 20px rgba(0, 255, 255, 0.3), 
                    0 0 30px rgba(255, 0, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# App Title and Brief Description
st.title("🎓 Campus Placement Prediction App")
st.write("""
    Welcome to the **Campus Placement Prediction App**! 🎉

    This app uses machine learning to predict a candidate's likelihood of placement based on various academic 
    and skill attributes. It’s a great tool for students to gauge their placement readiness and for recruiters 
    to pre-screen candidates.
""")

st.write("### 🔍 How to Use This App:")
st.write("""
1. **Input your details** in the sidebar by entering your academic and skill-related information.
2. **Choose a model** to train the app with different algorithms.
3. **Train the model** to see its performance on placement prediction.
4. **Make a Prediction**: Enter your details to get an estimation of your placement likelihood.
""")

def train_placement_model(model_choice):
    st.write("🔍 **Training Placement Prediction Model...**")

    # Load the dataset
    data_path = 'Placement_Data.csv'
    df = pd.read_csv(data_path)

    # Display column names to debug missing columns
    st.write("Dataset Columns:", df.columns.tolist())  # Print available columns for debugging

    # Preprocess the dataset
    st.write("🛠️ **Preprocessing Data...**")
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})  # Encode gender
    df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})  # Encode placement status
    df['workex'] = df['workex'].map({'Yes': 1, 'No': 0})  # Encode work experience

    # Only apply get_dummies to columns that exist in the DataFrame
    dummy_columns = ['degree_t', 'ssc_b', 'hsc_b', 'hsc_s']
    existing_columns = [col for col in dummy_columns if col in df.columns]
    
    if existing_columns:
        df = pd.get_dummies(df, columns=existing_columns, drop_first=True)
    else:
        st.write("⚠️ None of the dummy columns were found in the dataset.")

    # Define features (X) and target (y)
    X = df[['hsc_p', 'ssc_p', 'gender', 'workex', 'CGPA', 'Internships', 
            'Projects', 'Workshops/Certifications', 'AptitudeTestScore', 'SoftSkillsRating', 
            'AcademicConsistency', 'EngagementScore', 'PreparednessIndex', 'LeadershipScore']]
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
st.sidebar.header("📝 Input Features")
st.sidebar.write("📌 Please enter your academic and skill-related information in the fields below.")

# Model choice selection
st.sidebar.subheader("🛠️ Model Selection")
st.sidebar.write("Choose a machine learning model for training:")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"])

# Input fields for user to enter values
st.sidebar.subheader("🎓 Enter Your Academic Details")
st.sidebar.write("Provide your academic scores and details:")

ssc_b = st.sidebar.selectbox("📘 SSC Board", options=["Central", "Others"])
ssc_p = st.sidebar.slider("📈 SSC Percentage (%)", 0, 100, 50)

hsc_s = st.sidebar.selectbox("📚 HSC Specialization", options=["Commerce", "Science", "Arts"])
hsc_b = st.sidebar.selectbox("📙 HSC Board", options=["Central", "Others"])
hsc_p = st.sidebar.slider("📊 HSC Percentage (%)", 0, 100, 50)

gender = st.sidebar.selectbox("👤 Gender", options=["Male", "Female"])
work_experience = st.sidebar.selectbox("💼 Work Experience", options=["Yes", "No"])

# Additional inputs
st.sidebar.subheader("🛠️ Skills and Experiences")
st.sidebar.write("Provide details about your skills and relevant experiences:")


degree_t = st.sidebar.selectbox("🎓 Degree Type", options=["Sci&Tech", "Comm&Mgmt", "Others"])
CGPA = st.sidebar.slider("🎓 CGPA", 0.0, 10.0, 7.5)
internships = st.sidebar.number_input("💼 Number of Internships", min_value=0, value=0)
projects = st.sidebar.number_input("🔬 Number of Projects", min_value=0, value=0)
workshops_certifications = st.sidebar.number_input("📜 Number of Workshops/Certifications", min_value=0, value=0)
aptitude_test_score = st.sidebar.slider("🧠 Aptitude Test Score", 0, 100, 50)
soft_skills_rating = st.sidebar.slider("🤝 Soft Skills Rating (0-5)", 0, 5, 3)
academic_consistency = st.sidebar.slider("📐 Academic Consistency (-20 to 20)", -20.0, 20.0, 0.0)
engagement_score = st.sidebar.slider("📊 Engagement Score (0-10)", 0, 10, 5)
preparedness_index = st.sidebar.slider("📝 Preparedness Index (0-100)", 0, 100, 50)
leadership_score = st.sidebar.number_input("🌟 Leadership Score (0-5)", min_value=0, max_value=5, value=0)

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
