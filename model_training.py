import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data_path = "data/Placement_Data.csv"
df = pd.read_csv(data_path)

# Preprocessing: Convert categorical columns to numeric
df['gender'] = df['gender'].map({'M': 0, 'F': 1})
df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
df['workex'] = df['workex'].map({'Yes': 1, 'No': 0})  # Convert work experience to binary
df = pd.get_dummies(df, columns=['degree_t', 'ssc_b', 'hsc_b', 'hsc_s', 'ExtracurricularActivities', 'PlacementTraining',
                                 'TechnicalSkillLevel', 'FinancialSupportStatus', 'WillingnessToRelocate'])

# Define features and target variable
X = df[['hsc_p', 'ssc_p', 'gender', 'degree_p', 'workex', 'CGPA', 'Internships', 'Projects',
        'Workshops/Certifications', 'AptitudeTestScore', 'SoftSkillsRating', 'AcademicConsistency',
        'EngagementScore', 'PreparednessIndex', 'LeadershipScore']]
y = df['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'placement_model.joblib')

print("Model trained and saved as 'placement_model.joblib'.")
