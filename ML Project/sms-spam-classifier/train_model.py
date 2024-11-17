import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset with appropriate encoding
data = pd.read_csv('spam.csv', encoding='latin1')

# Step 2: Select and rename relevant columns
data = data[['v1', 'v2']]  # Keep only the columns we need
data.columns = ['label', 'text']  # Rename columns for clarity

# Step 3: Map labels to numeric values
data['label'] = data['label'].map({'spam': 1, 'ham': 0})  # Map 'spam' -> 1 and 'ham' -> 0

# Step 4: Extract features (text) and labels
X = data['text']  # Features
y = data['label']  # Labels

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Convert text data into TF-IDF features
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = tfidf.transform(X_test)  # Transform testing data

# Step 7: Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 9: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save the trained model pipeline
model_pipeline = {
    'model': model,
    'vectorizer': tfidf
}

# Save the pipeline as `sms_spam_classifier_pipeline.joblib`
joblib.dump(model_pipeline, 'sms_spam_classifier_pipeline.joblib')
print("Trained model saved as sms_spam_classifier_pipeline.joblib")
