## Libraries
import pandas as pd
import os
import nltk
import numpy as np
import re
import logging
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

## Dataset
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return the chat datasets with error handling."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, 'Dataset')
        
        chat_data_path = os.path.join(dataset_dir, 'dialog_talk_agent.xlsx')
        
        # Check for available training data
        train_files = [
            os.path.join(dataset_dir, 'model_data', 'train.json'),
            os.path.join(dataset_dir, 'model_data', 'dev.json')
        ]
        
        if not os.path.exists(chat_data_path):
            raise FileNotFoundError(f"File not found: {chat_data_path}")
            
        chat_data = pd.read_excel(chat_data_path)
        
        # Use dev.json as fallback since train.json doesn't exist
        chat_train_path = None
        for file_path in train_files:
            if os.path.exists(file_path):
                chat_train_path = file_path
                break
                
        if not chat_train_path:
            logger.warning("No training data found, using only dialog_talk_agent.xlsx")
            # Create empty DataFrame with required columns
            chat_train = pd.DataFrame(columns=['question', 'nq_answer'])
        else:
            chat_train = pd.read_json(chat_train_path)
            logger.info(f"Loaded training data from {chat_train_path}")
        
        # Process training data if available
        if not chat_train.empty:
            chat_train = chat_train.drop(columns=['viewed_doc_titles', 'used_queries', 'annotations', 'id', 'nq_doc_title'], errors='ignore')
            chat_train = chat_train.reindex(columns=['question', 'nq_answer'])
            chat_train = chat_train.rename(columns={'question': 'Context', 'nq_answer': 'Text Response'})
        
        logger.info("Datasets loaded successfully")
        return chat_data, chat_train
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

# Load datasets
chat_data, chat_train = load_datasets()

# removing brackets
def remove_brackets(text):
    new_text = str(text).replace('[', '') 
    new_text = str(new_text).replace(']', '')  
    return new_text

chat_train['Text Response'] = chat_train['Text Response'].apply(remove_brackets) 

# remove apostrophes 
def remove_first_and_last_character(text):
    return str(text)[1:-1]  

chat_train['Text Response'] = chat_train['Text Response'].apply(remove_first_and_last_character)


# data frame 
df = pd.DataFrame() 
column1 = [*chat_data['Context'].tolist(), *chat_train['Context'].tolist()]  
column2 = [*chat_data['Text Response'].tolist(), * chat_train['Text Response'].tolist()]
df.insert(0, 'Context', column1, True)  
df.insert(1, 'Text Response', column2, True)  

# flling missing values
df.ffill(axis = 0, inplace = True)

## Data Preprocessing
def cleaning(x):
    cleaned_array = list()
    for i in x:
        a = str(i).lower() 
        p = re.sub(r'[^a-z0-9]', ' ', a)  
        cleaned_array.append(p)  
    return cleaned_array

df.insert(1, 'Cleaned Context', cleaning(df['Context']), True)

# Data cleaning and lemmatization
def text_normalization(text):
    text = str(text).lower()  
    spl_char_text = re.sub(r'[^a-z]', ' ', text)  
    tokens = nltk.word_tokenize(spl_char_text)  
    lema = wordnet.WordNetLemmatizer()  
    tags_list = pos_tag(tokens, tagset = None) 
    lema_words = []
    for token, pos_token in tags_list:
        if pos_token.startswith('V'): 
            pos_val = 'v'
        elif pos_token.startswith('J'): 
            pos_val = 'a'
        elif pos_token.startswith('R'):  
            pos_val = 'r'
        else:  
            pos_val = 'n'
        lema_token = lema.lemmatize(token, pos_val)  
        lema_words.append(lema_token)  
    return " ".join(lema_words) 

normalized = df['Context'].apply(text_normalization)
df.insert(2, 'Normalized Context', normalized, True)

# removing stopwords
stop = stopwords.words('english')
def removeStopWords(text):
    Q = []
    s = text.split() 
    q = ''
    for w in s:  
        if w in stop:
            continue
        else:  
            Q.append(w)
        q = " ".join(Q)  
    return q

normalized_non_stopwords = df['Normalized Context'].apply(removeStopWords)
df.insert(3, 'Normalized and StopWords Removed', normalized_non_stopwords, True)

## Bag of words
cv = CountVectorizer() 
x_bow = cv.fit_transform(df['Normalized Context']).toarray() 

features_bow = cv.get_feature_names_out()  
df_bow = pd.DataFrame(x_bow, columns = features_bow)  

def chatbot_bow(question):
    tidy_question = text_normalization(removeStopWords(question))  
    cv_ = cv.transform([tidy_question]).toarray()  
    cos = 1- pairwise_distances(df_bow, cv_, metric = 'cosine') 
    index_value = cos.argmax()  
    return df['Text Response'].loc[index_value]