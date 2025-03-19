import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re

class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        self.classifier_nb = MultinomialNB()
        self.classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def train(self, X, y):
        print("\nTraining Data:")
        print("-" * 50)
        for i, (email, label) in enumerate(zip(X, y)):
            print(f"Email {i+1}: {email}")
            print(f"Label: {'SPAM' if label == 1 else 'NOT SPAM'}")
            print("-" * 50)
            
        # Preprocess the text data
        X_processed = [self.preprocess_text(text) for text in X]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # TF-IDF Vectorization
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train Naive Bayes
        self.classifier_nb.fit(X_train_tfidf, y_train)
        
        # Train Random Forest
        self.classifier_rf.fit(X_train_tfidf.toarray(), y_train)
        
        # Evaluate models
        nb_pred = self.classifier_nb.predict(X_test_tfidf)
        rf_pred = self.classifier_rf.predict(X_test_tfidf.toarray())
        
        print("\nModel Evaluation:")
        print("-" * 50)
        print("Naive Bayes Results:")
        print(classification_report(y_test, nb_pred))
        print("\nRandom Forest Results:")
        print(classification_report(y_test, rf_pred))
        
        return X_test, y_test
    
    def predict(self, text):
        # Preprocess new text
        processed_text = self.preprocess_text(text)
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Get predictions from both models
        nb_pred = self.classifier_nb.predict_proba(text_tfidf)[0]
        rf_pred = self.classifier_rf.predict_proba(text_tfidf.toarray())[0]
        
        # Ensemble prediction (average of both models)
        spam_prob = (nb_pred[1] + rf_pred[1]) / 2
        
        return {
            'is_spam': spam_prob > 0.5,
            'spam_probability': float(spam_prob),
            'naive_bayes_prob': float(nb_pred[1]),
            'random_forest_prob': float(rf_pred[1])
        }

# Example usage
if __name__ == "__main__":
    # Sample data (you should replace this with real email data)
    emails = [
        "Get rich quick! Buy now!",
        "Meeting at 3pm tomorrow",
        "CONGRATULATIONS! You've won $1,000,000!",
        "Please review the project proposal",
        "100% FREE - Limited time offer!!!",
        "Dear colleague, please find attached the meeting minutes",
        "URGENT: Your account has been suspended",
        "Can we discuss the project timeline?",
        "Claim your prize now! Don't miss out!",
        "Schedule: Team sync at 2 PM"
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for non-spam
    
    print("\n=== Spam Email Classifier ===")
    print("\nInitializing and training the classifier...")
    
    # Create and train the classifier
    classifier = SpamClassifier()
    classifier.train(emails, labels)
    
    # Test with new emails
    test_emails = [
        "Hello, can we schedule a meeting for tomorrow?",
        "WINNER! WINNER! Claim your prize NOW!!!",
        "Please find attached the quarterly report"
    ]
    
    print("\nTesting New Emails:")
    print("=" * 50)
    for email in test_emails:
        result = classifier.predict(email)
        print(f"\nEmail: {email}")
        print("-" * 30)
        print(f"Result: {'SPAM' if result['is_spam'] else 'NOT SPAM'}")
        print(f"Spam Probability: {result['spam_probability']:.2f}")
        print(f"Individual Model Probabilities:")
        print(f"- Naive Bayes: {result['naive_bayes_prob']:.2f}")
        print(f"- Random Forest: {result['random_forest_prob']:.2f}")
        print("=" * 50)
