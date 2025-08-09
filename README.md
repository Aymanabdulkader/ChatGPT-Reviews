# ChatGPT-Reviews

import pandas as pd  

df = pd.read_csv("/content/drive/MyDrive/New folder/chatgpt_reviews.csv")

df

df.columns

print(df.columns)

missing_values = df.isnull().sum()
print("Missing values before handling:")
display(missing_values)

df.dropna(subset=['Review'], inplace=True)
print("\nMissing values after handling:")
display(df.isnull().sum())

df['Review Date'] = pd.to_datetime(df['Review Date'])
print("Data type of 'Review Date' after conversion:")
display(df['Review Date'].dtype)

print("Number of duplicate rows before removal:", df.duplicated().sum())

df.drop_duplicates(inplace=True)
print("Number of duplicate rows after removal:", df.duplicated().sum())

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['cleaned_review'] = df['Review'].apply(clean_text)
display(df[['Review', 'cleaned_review']].head())

from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['cleaned_review'].apply(analyzer.polarity_scores)
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

display(df[['cleaned_review', 'sentiment_scores', 'compound']].head())

def categorize_sentiment(compound_score):
    if compound_score > 0.05:
        return 'Positive'
    elif compound_score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['compound'].apply(categorize_sentiment)
display(df[['Review', 'compound', 'sentiment_category']].head())

sentiment_counts = df['sentiment_category'].value_counts()
print("Distribution of Sentiment Categories (Counts):")
display(sentiment_counts)

sentiment_percentages = df['sentiment_category'].value_counts(normalize=True) * 100
print("\nDistribution of Sentiment Categories (Percentages):")
display(sentiment_percentages.round(2))

monthly_sentiment = df.set_index('Review Date').resample('M')['compound'].mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sentiment.index, monthly_sentiment.values)
plt.xlabel('Review Date')
plt.ylabel('Average Compound Sentiment Score')
plt.title('Average Sentiment Over Time')
plt.grid(True)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0f27f9a0-8d0c-4bd7-8b81-8256c6ab3ea5" />

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

negative_reviews_df = df[df['sentiment_category'] == 'Negative']

negative_text = " ".join(negative_reviews_df['cleaned_review'].tolist())

vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = vectorizer.fit_transform([negative_text])

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A[0]

tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))

sorted_tfidf_scores = pd.Series(tfidf_scores_dict).sort_values(ascending=False)

top_n = 20
print(f"Top {top_n} terms in negative reviews based on TF-IDF scores:")
display(sorted_tfidf_scores.head(top_n))

plt.figure(figsize=(10, 8))
sorted_tfidf_scores.head(top_n).plot(kind='barh')
plt.xlabel('TF-IDF Score')
plt.title(f'Top {top_n} Terms in Negative Reviews')
plt.gca().invert_yaxis()
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2ff947f1-6064-43ca-ada3-dd98ae423203" />

# We already have the top terms from the previous step in `sorted_tfidf_scores`
# Let's use the top N terms as identified previously
top_terms = sorted_tfidf_scores.head(top_n).index.tolist()

# Function to count occurrences of terms in a review
def count_terms(review, terms):
    count = 0
    for term in terms:
        if term in review:
            count += 1
    return count

# Apply the function to count occurrences of top terms in each negative review
negative_reviews_df['issue_count'] = negative_reviews_df['cleaned_review'].apply(lambda x: count_terms(x, top_terms))

# Filter for reviews that contain at least one of the top terms
reviews_with_issues = negative_reviews_df[negative_reviews_df['issue_count'] > 0]

print(f"Number of negative reviews containing at least one of the top {top_n} terms:")
display(reviews_with_issues.shape[0])

# Flatten the list of words in negative reviews, keeping only the top terms
issue_terms_list = [word for review in reviews_with_issues['cleaned_review'] for word in review.split() if word in top_terms]

print("Sample of extracted issue terms:")
display(issue_terms_list[:20])

from collections import Counter

# Count the frequency of each issue term
issue_term_counts = Counter(issue_terms_list)

# Display the most common issue terms
most_common_issues = pd.Series(issue_term_counts).sort_values(ascending=False)

print("Most common issue terms in negative reviews:")
display(most_common_issues.head(top_n))

plt.figure(figsize=(10, 8))
most_common_issues.head(top_n).plot(kind='barh')
plt.xlabel('Frequency')
plt.title(f'Most Common Issue Terms in Negative Reviews (Top {top_n})')
plt.gca().invert_yaxis()
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f95a6a9c-cc30-4de6-a6c3-e63873772659" />

import matplotlib.pyplot as plt

# Group by month and calculate the average compound sentiment
monthly_sentiment = df.set_index('Review Date').resample('M')['compound'].mean()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_sentiment.index, monthly_sentiment.values)
plt.xlabel('Review Date')
plt.ylabel('Average Compound Sentiment Score')
plt.title('Average Sentiment Over Time')
plt.grid(True)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a320b687-1747-4f9a-ab0c-ae6b6e8b41b2" />

# We already have the top terms from the previous step in `sorted_tfidf_scores`
# Let's use the top N terms as identified previously
top_terms = sorted_tfidf_scores.head(top_n).index.tolist()

# Function to count occurrences of terms in a review
def count_terms(review, terms):
    count = 0
    for term in terms:
        if term in review:
            count += 1
    return count

# Apply the function to count occurrences of top terms in each negative review
negative_reviews_df['issue_count'] = negative_reviews_df['cleaned_review'].apply(lambda x: count_terms(x, top_terms))

# Filter for reviews that contain at least one of the top terms
reviews_with_issues = negative_reviews_df[negative_reviews_df['issue_count'] > 0]

print(f"Number of negative reviews containing at least one of the top {top_n} terms:")
display(reviews_with_issues.shape[0])

# Flatten the list of words in negative reviews, keeping only the top terms
issue_terms_list = [word for review in reviews_with_issues['cleaned_review'] for word in review.split() if word in top_terms]

print("Sample of extracted issue terms:")
display(issue_terms_list[:20])

from collections import Counter

# Count the frequency of each issue term
issue_term_counts = Counter(issue_terms_list)

# Display the most common issue terms
most_common_issues = pd.Series(issue_term_counts).sort_values(ascending=False)

print("Most common issue terms in negative reviews:")
display(most_common_issues.head(top_n))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
most_common_issues.head(top_n).plot(kind='barh')
plt.xlabel('Frequency')
plt.title(f'Most Common Issue Terms in Negative Reviews (Top {top_n})')
plt.gca().invert_yaxis()
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2a3fa823-b6e5-448d-84d6-7a727160e8b6" />

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

positive_reviews_df = df[df['sentiment_category'] == 'Positive']

positive_text = " ".join(positive_reviews_df['cleaned_review'].tolist())

vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = vectorizer.fit_transform([positive_text])

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A[0]

tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))

sorted_tfidf_scores_positive = pd.Series(tfidf_scores_dict).sort_values(ascending=False)

top_n = 20
print(f"Top {top_n} terms in positive reviews based on TF-IDF scores:")
display(sorted_tfidf_scores_positive.head(top_n))

plt.figure(figsize=(10, 8))
sorted_tfidf_scores_positive.head(top_n).plot(kind='barh')
plt.xlabel('TF-IDF Score')
plt.title(f'Top {top_n} Terms in Positive Reviews')
plt.gca().invert_yaxis()
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a1b3a94e-ce71-48d4-ab16-d1b6f353777d" />

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

neutral_reviews_df = df[df['sentiment_category'] == 'Neutral']

neutral_text = " ".join(neutral_reviews_df['cleaned_review'].tolist())

vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = vectorizer.fit_transform([neutral_text])

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A[0]

tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))

sorted_tfidf_scores_neutral = pd.Series(tfidf_scores_dict).sort_values(ascending=False)

top_n = 20
print(f"Top {top_n} terms in neutral reviews based on TF-IDF scores:")
display(sorted_tfidf_scores_neutral.head(top_n))

plt.figure(figsize=(10, 8))
sorted_tfidf_scores_neutral.head(top_n).plot(kind='barh')
plt.xlabel('TF-IDF Score')
plt.title(f'Top {top_n} Terms in Neutral Reviews')
plt.gca().invert_yaxis()
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/994902fa-c42d-456d-a79d-d0fcd042955e" />






