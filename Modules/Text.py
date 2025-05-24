from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
import string
import re
import pandas as pd
from joblib import load

# Custom tokenizer with lemmatization and cleaning
def custom_tokenizer(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    text = contractions.fix(text) # Expand contractions
    text = str(text).lower() # Convert to lowercase
    
    # Url Handling
    url_match  = re.search(r'https?://(?:www\.)?([^/]+)', text)
    domain = ''
    if url_match:
        domain = url_match.group(1)
        domain = ' '.join(domain.split('.'))
        # Extract the path and split into parts
        path = re.sub(r'https?://[^/]+/', '', text)
        words = re.split(r'[/\-.#]', path)
        words.insert(1, domain)
        text = ' '.join(words)

    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize and filter
    tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 2]

    tokenized_text = ' '.join(tokens)
    return tokenized_text

def tf_idf(df_merged, folder_path):
    text_columns = [
        'Acquisition_News',
        'Acquisition_News_Link',
        'Acquiring_Tagline',
        'Acquiring_Description',
        'Acquired_Tagline',
        'Acquired_Description'
    ]

    # Preprocess text data - fill NA and combine relevant columns
    df_merged[text_columns] = df_merged[text_columns].fillna('')

    # Create combined text features that might work better together
    df_merged['Acquiring_Text'] = df_merged['Acquiring_Tagline'] + " " + df_merged['Acquiring_Description']
    df_merged['Acquired_Text'] = df_merged['Acquired_Tagline'] + " " + df_merged['Acquired_Description']
    df_merged['News_Text'] = df_merged['Acquisition_News'] + " " + df_merged['Acquisition_News_Link']

    df_merged.drop(columns=text_columns, inplace=True)
    text_columns = ['Acquiring_Text', 'Acquired_Text', 'News_Text']

    for column in text_columns:
        print(column, ":-")
        print("Text_Before:", df_merged[column].iloc[0])
        df_merged[column] =  df_merged[column].apply(custom_tokenizer)
        print("Text_After:", df_merged[column].iloc[0])
        count = df_merged[column].apply(lambda x: len(x.split(' ')))
        average_count = int(count.sum() / count.shape[0])
        print("average number of tokens:", average_count, '\n')
    
    x = 0
    for column in text_columns:
        # Load the pre-trained TF-IDF vectorizer
        tfidf = load(folder_path + 'Models\\' + f'tfidf_vectorizer_{x}.joblib')

        # Get the original feature names from the vectorizer
        original_features = tfidf.get_feature_names_out()

        # Transform with the pre-trained vectorizer
        tfidf_matrix = tfidf.transform(df_merged[column].fillna(''))
        
        # Create DataFrame using the original feature names
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"{column}_{feat}" for feat in original_features],
            index=df_merged.index
        )
        
        # Join back with original DataFrame
        df_merged = pd.concat([df_merged, tfidf_df], axis=1)
        x += 1

    df_merged.drop(columns=text_columns, inplace=True)
    return df_merged