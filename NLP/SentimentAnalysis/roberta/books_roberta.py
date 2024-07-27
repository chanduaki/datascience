import pandas as pd
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Step 1: Load Data
file_path = 'merged_data.csv'
df = pd.read_csv(file_path)

# Step 2: Preprocess Text Data
def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove punctuation
        words = word_tokenize(text)  # Tokenize the text
        filtered_words = [word.lower() for word in words]  # Convert to lowercase
        cleaned_text = ' '.join(filtered_words)  # Join the filtered words back into a string
        return cleaned_text
    else:
        return ''  # If not a string, return an empty string or handle as appropriate

df['cleaned_feedback'] = df['review description'].apply(clean_text)
print(df['cleaned_feedback'])

# Step 3: Load RoBERTa Model
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
print("--- Model Loaded ---")

# Step 4: Define Sentiment Analysis Function
def sentiment_score(review):
    # Tokenize the input review with padding and truncation
    tokens = tokenizer.encode(review, return_tensors='pt', truncation=True, padding=True, max_length=512)
    result = model(tokens)
    probabilities = softmax(result.logits, dim=1).detach().numpy()[0]
    label_mapping = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
    scores = {label_mapping[model.config.id2label[i]]: prob for i, prob in enumerate(probabilities)}
    return scores


# Step 5: Define function to get the overall sentiment label and score
def get_overall_sentiment(scores):
    overall_sentiment_label = max(scores, key=scores.get)
    weights = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    overall_sentiment_score = sum(score * weights[label] for label, score in scores.items())
    return overall_sentiment_label, overall_sentiment_score

# Step 6: Analyze Sentiments, Calculate Overall Sentiment, and Store Results in DataFrame
df['Positive'] = 0.0
df['Negative'] = 0.0
df['Neutral'] = 0.0
df['Overall_Sentiment_Label'] = ''
df['Overall_Sentiment_Score'] = 0.0

print("Started: Calculating scores")
for index, row in df.iterrows():
    comment = row['cleaned_feedback']
    scores = sentiment_score(comment)
    print(f"Row number: {index}")

    df.at[index, 'Positive'] = scores.get('Positive', 0)
    df.at[index, 'Negative'] = scores.get('Negative', 0)
    df.at[index, 'Neutral'] = scores.get('Neutral', 0)

    overall_label, overall_score = get_overall_sentiment(scores)
    df.at[index, 'Overall_Sentiment_Label'] = overall_label
    df.at[index, 'Overall_Sentiment_Score'] = overall_score

    # Save the DataFrame in each iteration
    df.to_csv('books_data_with_sentiment_score_in_progress.csv', index=False)

print("Ended: Calculating scores")

# Step 7: Save the final DataFrame
df.to_csv('books_data_with_sentiment_score_final.csv', index=False)  # Save the final DataFrame to a new CSV file

# Step 8: Display Results
print(df.head())  # Display the first few rows of the updated DataFrame
