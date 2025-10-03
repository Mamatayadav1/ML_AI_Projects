# RNN Sentiment Analysis Notebook
# Dataset: Sentiment Analysis Dataset from GitHub
# Dataset URL: https://raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("="*60)
print("RNN SENTIMENT ANALYSIS PROJECT")
print("="*60)
print("Dataset Source: https://raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv")
print("="*60)

# Load the dataset
print("\n1. Loading Dataset...")
url = "https://raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv"
data = pd.read_csv(url)

print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print("\nFirst few rows:")
print(data.head())

# Data exploration
print("\n2. Data Exploration...")
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nSentiment distribution:\n{data['Sentiment'].value_counts()}")

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
data['Sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Text preprocessing
print("\n3. Text Preprocessing...")
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(words)

# Apply preprocessing
data['clean_text'] = data['SentimentText'].apply(clean_text)

# Remove empty texts
data = data[data['clean_text'].str.len() > 0]
print(f"Dataset shape after cleaning: {data.shape}")

# Encode labels
print("\n4. Label Encoding...")
le = LabelEncoder()
data['sentiment_encoded'] = le.fit_transform(data['Sentiment'])
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Prepare features and target
X = data['clean_text'].values
y = data['sentiment_encoded'].values

# Split the data
print("\n5. Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Tokenization and padding
print("\n6. Text Tokenization...")
vocab_size = 10000
max_length = 100
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

print(f"Training sequences shape: {X_train_pad.shape}")
print(f"Validation sequences shape: {X_val_pad.shape}")
print(f"Test sequences shape: {X_test_pad.shape}")

# Build RNN model
print("\n7. Building RNN Model...")
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    SimpleRNN(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Training callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.0001,
    verbose=1
)

# Train the model
print("\n8. Training the Model...")
history = model.fit(
    X_train_pad, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val_pad, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
print("\n9. Training History...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Model evaluation
print("\n10. Model Evaluation...")

# Validation set evaluation
val_loss, val_accuracy = model.evaluate(X_val_pad, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test set evaluation
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification report
print("\n11. Classification Report...")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Test with custom examples
print("\n12. Testing with Custom Examples...")
test_texts = [
    "I love this product! It's amazing!",
    "This is terrible, I hate it",
    "The weather is nice today",
    "I'm feeling sad and depressed",
    "Great job! Well done!",
    "This is the worst experience ever"
]

print("\nPredicting sentiments for custom examples:")
for text in test_texts:
    # Preprocess the text
    clean_text = clean_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded)[0][0]
    predicted_sentiment = le.inverse_transform([int(prediction > 0.5)])[0]
    
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {predicted_sentiment} (confidence: {prediction:.3f})")
    print("-" * 50)

# Performance summary
print("\n13. Performance Summary...")
print("="*60)
print(f"FINAL RESULTS:")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print("="*60)

# Check if accuracy is above 90%
if test_accuracy > 0.9:
    print("✅ SUCCESS: Model achieved >90% accuracy!")
else:
    print("⚠️  Note: Model accuracy is below 90%. Consider:")
    print("- Increasing model complexity")
    print("- Adding more training data")
    print("- Tuning hyperparameters")
    print("- Using pre-trained embeddings")

print("\nModel training completed successfully!")
print("You can now use this model for sentiment analysis tasks.")