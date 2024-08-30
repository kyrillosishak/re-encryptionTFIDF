:::{.cell .markdown}

# Introduction

:::

:::{.cell .markdown}

Encryption algorithms play a pivotal role in safeguarding sensitive information. However, even with sophisticated encryption techniques, the risk of data leakage remains a critical concern. This notebook aims to reproduce the study presented in "Exploring Data Leakage in Encrypted Payload Using Supervised Machine Learning" by Amir Khaleghi Moghaddam and Nur Zincir-Heywood. The paper investigates whether data leakage occurs when encrypted data is analyzed using supervised machine learning techniques. It evaluates four different encryption algorithms and their susceptibility to data leakage by applying various supervised learning methods to encrypted payloads.

We aim to investigate whether the classification performance of the text data changes after encryption. Specifically, we want to determine if the classification accuracy remains the same or deteriorates once the text has been encrypted. This involves evaluating whether the text's semantic information is preserved sufficiently to ensure consistent classification results, or if encryption introduces distortions that negatively impact the model's performance.

:::

:::{.cell .markdown}
## In This Notebook We Will:
1. Replicate the Data Encryption and Classification Experiments: We will encrypt plaintext messages using different encryption algorithms and key sizes, then use supervised learning techniques to classify the encrypted messages by their topics.

2. Explore the Impact of Encryption Strength on Data Leakage: We will analyze how varying encryption algorithms affect the extent of data leakage, using metrics to evaluate the effectiveness of each encryption method.

3. Evaluate Supervised Learning Techniques: Various supervised learning algorithms will be employed to determine their ability to classify encrypted messages and reveal any potential data leakage.

::: 

:::{.cell .markdown}
## Downloading the data
:::

:::{.cell .markdown}

To download data from kaggle : 
- Get Your Kaggle API Key
  1. Go to the Kaggle website and log in to your account.
  2. Navigate to your profile by clicking on your profile picture in the top right corner and selecting "Account".
  3. scroll down to the "API" section and click on "Create New API Token". This will download a kaggle.json file to your computer, containing your API credentials.
- Run API command below

:::

:::{.cell .code}
```python
import json
import os

# Read your kaggle.json file
with open('kaggle.json', 'r') as f:
    kaggle_creds = json.load(f)

# Set environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
os.environ['KAGGLE_KEY'] = kaggle_creds['key']

```
::: 

:::{.cell .code}
```python
!kaggle datasets download -d activegalaxy/isis-related-tweets
!unzip -q /content/isis-related-tweets.zip
```
::

:::{.cell .code}
```python
!pip install pycryptodome
```
:::

:::{.cell .code}
```python
import numpy as np
import pandas as pd
import re
import string
import base64
from Crypto.Cipher import DES, DES3, AES
from Crypto.Util.Padding import pad
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
```
:::

:::{.cell .markdown}
## Data cleaning

Although the authors indicated that no preprocessing such as stemming or lowercasing was applied to the data, we have removed unnecessary links and numbers.

:::

:::{.cell .code}
```python
# Function to clean text
def clean_text(text):
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove new lines
    text = text.lower()  # Convert to lowercase
    return text
```
:::

:::{.cell .markdown}
## Loading data and defining encryption algorithms
:::

:::{.cell .code}
```python
def load_data():
    # Load datasets with specified encoding
    about_isis = pd.read_csv('AboutIsis.csv')
    isis_fanboy = pd.read_csv('IsisFanboy.csv', encoding='ISO-8859-1')

    # Assign labels: 0 for AboutIsis, 1 for IsisFanboy
    about_isis['label'] = 0
    isis_fanboy['label'] = 1

    # Take 15000 samples from each dataframe
    about_isis = about_isis.sample(n=15000, random_state=50)
    isis_fanboy = isis_fanboy.sample(n=15000, random_state=50)

    # Combine the text and labels
    texts = pd.concat([about_isis['tweets'], isis_fanboy['tweets']]).values
    labels = pd.concat([about_isis['label'], isis_fanboy['label']]).values

    # Convert non-bytes-like objects to strings, handling floats
    texts = [str(text).encode('utf-8') if not isinstance(text, bytes) else text for text in texts]

    # Clean the text
    cleaned_texts = [clean_text(text.decode('utf-8')) for text in texts]
    return cleaned_texts, labels
```
:::

:::{.cell .code}
```python
# Function to encrypt data
def encrypt_text(text, key, algorithm='DES', mode = 'ECB'):
    if algorithm == 'DES':
        if mode == 'ECB':
            cipher = DES.new(key, DES.MODE_ECB)
        elif mode == 'CBC':
            cipher = DES.new(key, DES.MODE_CBC, iv=b'00000000')  
    elif algorithm == '3DES':
        if mode == 'ECB':
            cipher = DES3.new(key, DES3.MODE_ECB)
        elif mode == 'CBC':
            cipher = DES3.new(key, DES3.MODE_CBC, iv=b'00000000') 
    elif algorithm == 'AES':
        if mode == 'ECB':
            cipher = AES.new(key, AES.MODE_ECB)
        elif mode == 'CBC':
            cipher = AES.new(key, AES.MODE_CBC, iv=b'0000000000000000')  
    encrypted = cipher.encrypt(pad(text.encode(), cipher.block_size))
    return base64.b64encode(encrypted).decode()
```
:::

:::{.cell .markdown}
## Training and printing results
:::

:::{.cell .code}
```python
def train(texts, labels, key, algorithm, mode):
    # Encrypt all texts
    encrypted_texts = [encrypt_text(text, key, algorithm, mode) for text in texts]

    # TF-IDF on all encrypted texts
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), max_features=5000)
    X = vectorizer.fit_transform(encrypted_texts)

    # Evaluate using cross-validation
    results = {}
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Linear SVC': LinearSVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Multinomial Naive Bayes': MultinomialNB()
    }

    for name, model in models.items():
        print(f"Evaluating encryption type : {algorithm},{mode} Model : {name} ...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, y_pred_train, average='binary')

        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        results[name] = {
            'train' : {
                'Precision': precision_train,
                'Recall': recall_train,
                'F1 Score': f1_train
            },
            'test' : {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
        }

    return results
```
:::

:::{.cell .code}
```python
texts, labels = load_data()
des_key = b'12345678'  # 8 bytes key for DES
tdes_key_128 = b'1234567890ABCDEF'  # 16 bytes key for 3DES
tdes_key_192 = b'1234567890ABCDEF12345678'  # 24 bytes key for 3DES
aes_key_128 = b'1234567890ABCDEF'  # 16 bytes key for AES
aes_key_192 = b'1234567890ABCDEF12345678'  # 24 bytes key for AES
aes_key_256 = b'1234567890ABCDEF1234567890ABCDEF'  # 32 bytes key for AES

# Assume `train` is a function that trains the model and returns the metrics
results = {
    'DES (64-bit, ECB)': train(texts, labels, des_key, 'DES', 'ECB'),
    'DES (64-bit, CBC)': train(texts, labels, des_key, 'DES', 'CBC'),
    
    '3DES (128-bit, ECB)': train(texts, labels, tdes_key_128, '3DES', 'ECB'),
    '3DES (128-bit, CBC)': train(texts, labels, tdes_key_128, '3DES', 'CBC'),
    
    '3DES (192-bit, ECB)': train(texts, labels, tdes_key_192, '3DES', 'ECB'),
    '3DES (192-bit, CBC)': train(texts, labels, tdes_key_192, '3DES', 'CBC'),
    
    'AES (128-bit, ECB)': train(texts, labels, aes_key_128, 'AES', 'ECB'),
    'AES (128-bit, CBC)': train(texts, labels, aes_key_128, 'AES', 'CBC'),
    
    'AES (192-bit, ECB)': train(texts, labels, aes_key_192, 'AES', 'ECB'),
    'AES (192-bit, CBC)': train(texts, labels, aes_key_192, 'AES', 'CBC'),
    
    'AES (256-bit, ECB)': train(texts, labels, aes_key_256, 'AES', 'ECB'),
    'AES (256-bit, CBC)': train(texts, labels, aes_key_256, 'AES', 'CBC'),
}
```
:::

:::{.cell .code}
```python
def print_results(results):
    print("+"*49 +" Train " +"+"*49)
    # Iterate over each encryption method
    for encryption, classifiers in results.items():
        print(f"Results for {encryption}:\n")
        
        # Print header for classifiers
        print(f"{'Metric':<20} {'Logistic Regression':<23} {'Linear SVC':<18} {'Random Forest':<16} {'Multinomial Naive Bayes':<17}")
        print("=" * 104)
        
        # Print Precision, Recall, F1 Score for each classifier
        
        for metric in ['Precision', 'Recall', 'F1 Score']:
            ss = ""
            for clf, metrics in classifiers.items():
                train_metric = metrics['train'][metric]
                ss += f"{train_metric:<20.4f}"
            print(f"{metric:<25} {ss}")
        
        print("\n" + "-" * 104 + "\n")

    print("+"*49 +" Test " +"+"*49)
    # Iterate over each encryption method
    for encryption, classifiers in results.items():
        print(f"Results for {encryption}:\n")
        
        # Print header for classifiers
        print(f"{'Metric':<20} {'Logistic Regression':<23} {'Linear SVC':<18} {'Random Forest':<16} {'Multinomial Naive Bayes':<17}")
        print("=" * 104)
        
        # Print Precision, Recall, F1 Score for each classifier
        
        for metric in ['Precision', 'Recall', 'F1 Score']:
            ss = ""
            for clf, metrics in classifiers.items():
                test_metric = metrics['test'][metric]
                ss += f"{test_metric:<20.4f}"
            print(f"{metric:<25} {ss}")
        
        print("\n" + "-" * 104 + "\n")

# Example usage with your data
print_results(results)

```
:::