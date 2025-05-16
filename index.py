import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Step 1: Load the dataset
df = pd.read_csv("US_Accidents_March23.csv")
print("Initial shape of dataset:", df.shape)

# Step 2: Select relevant columns
selected_columns = [
    'Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)',
    'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Weather_Condition', 'Sunrise_Sunset'
]

df = df[selected_columns]
df.dropna(inplace=True)
print("Shape after dropping missing values:", df.shape)

# Step 3: Encode categorical features
label_encoder = LabelEncoder()
df['Weather_Condition'] = label_encoder.fit_transform(df['Weather_Condition'])
df['Sunrise_Sunset'] = label_encoder.fit_transform(df['Sunrise_Sunset'])

# Step 4: Define features and target
X = df.drop('Severity', axis=1)
y = df['Severity']

# **FIX**: Encode the target variable (Severity) to be zero-indexed
y = label_encoder.fit_transform(y)

# Step 5: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance for Traffic Accident Severity Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
