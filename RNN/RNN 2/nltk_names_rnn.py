# NLTK Names Corpus RNN Model for Gender Classification
# Dataset: https://www.nltk.org/nltk_data/

import nltk
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download NLTK names corpus
print("Downloading NLTK names corpus...")
nltk.download('names')

# Load the names corpus
from nltk.corpus import names

# Prepare the data
print("Loading and preparing data...")
male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Create dataset
data = []
labels = []

# Add male names (label 0)
data.extend([name.lower() for name in male_names])
labels.extend([0] * len(male_names))

# Add female names (label 1)
data.extend([name.lower() for name in female_names])
labels.extend([1] * len(female_names))

print(f"Total names: {len(data)}")
print(f"Male names: {len(male_names)}")
print(f"Female names: {len(female_names)}")

# Create DataFrame
df = pd.DataFrame({
    'name': data,
    'gender': labels
})

# Display sample data
print("\nSample data:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"Gender distribution:\n{df['gender'].value_counts()}")

# Character-level tokenization
print("\nPreparing character-level features...")
all_chars = set(''.join(data))
char_to_idx = {char: idx+1 for idx, char in enumerate(sorted(all_chars))}
char_to_idx['<PAD>'] = 0
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

vocab_size = len(char_to_idx)
max_length = max(len(name) for name in data)

print(f"Vocabulary size: {vocab_size}")
print(f"Maximum name length: {max_length}")

# Convert names to sequences
sequences = []
for name in data:
    seq = [char_to_idx[char] for char in name]
    sequences.append(seq)

# Pad sequences
X = pad_sequences(sequences, maxlen=max_length, padding='post', value=0)
y = np.array(labels)

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Build RNN model
print("\nBuilding RNN model...")
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))

# Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot training history
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Test with sample names
print("\nTesting with sample names:")
sample_names = ['john', 'mary', 'david', 'sarah', 'michael', 'emma', 'robert', 'lisa']

for name in sample_names:
    # Convert name to sequence
    seq = [char_to_idx.get(char, 0) for char in name.lower()]
    seq_padded = pad_sequences([seq], maxlen=max_length, padding='post', value=0)
    
    # Predict
    pred_prob = model.predict(seq_padded, verbose=0)[0][0]
    pred_gender = 'Female' if pred_prob > 0.5 else 'Male'
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    
    print(f"Name: {name.capitalize()} -> {pred_gender} (Confidence: {confidence:.3f})")

# Additional model metrics
print(f"\nModel Performance Summary:")
print(f"Training Accuracy: {max(history.history['accuracy']):.4f}")
print(f"Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Total Parameters: {model.count_params()}")

# Feature importance - Last character analysis
print("\nAnalyzing last character patterns...")
last_char_male = {}
last_char_female = {}

for i, name in enumerate(data):
    last_char = name[-1]
    if labels[i] == 0:  # Male
        last_char_male[last_char] = last_char_male.get(last_char, 0) + 1
    else:  # Female
        last_char_female[last_char] = last_char_female.get(last_char, 0) + 1

print("\nTop 10 last characters for male names:")
male_sorted = sorted(last_char_male.items(), key=lambda x: x[1], reverse=True)[:10]
for char, count in male_sorted:
    print(f"'{char}': {count}")

print("\nTop 10 last characters for female names:")
female_sorted = sorted(last_char_female.items(), key=lambda x: x[1], reverse=True)[:10]
for char, count in female_sorted:
    print(f"'{char}': {count}")

# Save model
model.save('names_gender_rnn_model.h5')
print("\nModel saved as 'names_gender_rnn_model.h5'")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("="*50)