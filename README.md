# ChatGPT-Reviews

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/New folder/chatgpt_reviews.csv")
df
df.columns
df.head()

import re
import nltk
from nltk.corpus import stopwords
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):  # Check if the value is a string
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    df['cleaned_review'] = df['Review'].apply(clean_text)

def clean_text(text):
    """
    Cleans a text string by converting to lowercase, removing special characters,
    and removing stop words. Handles non-string inputs by returning an empty string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_review'] = df['Review'].apply(clean_text)
display(df[['Review', 'cleaned_review']].head())

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

sid = SentimentIntensityAnalyzer()

def get_vader_sentiment_scores(review):
    """
    Returns VADER sentiment scores for a given text review.
    """
    if not isinstance(review, str):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return sid.polarity_scores(review)

sentiment_scores = df['cleaned_review'].apply(get_vader_sentiment_scores)

df['compound_score'] = sentiment_scores.apply(lambda x: x['compound'])
df['negative_score'] = sentiment_scores.apply(lambda x: x['neg'])
df['neutral_score'] = sentiment_scores.apply(lambda x: x['neu'])
df['positive_score'] = sentiment_scores.apply(lambda x: x['pos'])

display(df[['cleaned_review', 'compound_score', 'negative_score', 'neutral_score', 'positive_score']].head())

print("Descriptive Statistics for Sentiment Scores:")
print(df[['compound_score', 'negative_score', 'neutral_score', 'positive_score']].describe())

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(df['compound_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Compound Sentiment Scores')
plt.xlabel('Compound Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(df['negative_score'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Negative Sentiment Scores')
plt.xlabel('Negative Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(df['neutral_score'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Neutral Sentiment Scores')
plt.xlabel('Neutral Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(df['positive_score'], bins=20, color='gold', edgecolor='black')
plt.title('Distribution of Positive Sentiment Scores')
plt.xlabel('Positive Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(column=['compound_score'], by='Ratings')
plt.title('Compound Sentiment Score by Ratings')
plt.xlabel('Ratings')
plt.ylabel('Compound Score')
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(column=['positive_score'], by='Ratings')
plt.title('Positive Sentiment Score by Ratings')
plt.xlabel('Ratings')
plt.ylabel('Positive Score')
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(column=['negative_score'], by='Ratings')
plt.title('Negative Sentiment Score by Ratings')
plt.xlabel('Ratings')
plt.ylabel('Negative Score')
plt.show()

plt.figure(figsize=(10, 6))
df.boxplot(column=['neutral_score'], by='Ratings')
plt.title('Neutral Sentiment Score by Ratings')
plt.xlabel('Ratings')
plt.ylabel('Neutral Score')
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/92aca7d6-aa9e-4290-b865-aec768d505ab" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/73081c32-4f40-452d-94d2-983565219364" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6e5832e8-e952-4735-b634-334465e5a95c" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/03600f7b-50e8-49fd-b588-67b1452eccac" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f905bb6f-5603-482d-87b3-bbc2317249cf" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/41297c65-0ab7-442f-a79b-f940049632c7" />

negative_reviews_df = df[df['compound_score'] < 0]
display(negative_reviews_df.head())

from collections import Counter
import itertools

# Calculate word frequencies
all_words = list(itertools.chain.from_iterable(negative_reviews_df['cleaned_review'].str.split()))
word_counts = Counter(all_words)

# Identify common and rare words (thresholds can be adjusted)
total_words = len(all_words)
common_threshold = 0.01 # words appearing in more than 1% of reviews
rare_threshold = 5      # words appearing less than 5 times

common_words = {word for word, count in word_counts.items() if count / len(negative_reviews_df) > common_threshold}
rare_words = {word for word, count in word_counts.items() if count < rare_threshold}

# Function to remove common and rare words
def remove_common_rare(text):
    return ' '.join([word for word in text.split() if word not in common_words and word not in rare_words])

# Apply the function to create the new column
negative_reviews_df['further_cleaned_review'] = negative_reviews_df['cleaned_review'].apply(remove_common_rare)

# Display the results
display(negative_reviews_df[['cleaned_review', 'further_cleaned_review']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Fit and transform the 'further_cleaned_review' column
tfidf_matrix = tfidf_vectorizer.fit_transform(negative_reviews_df['further_cleaned_review'])

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Calculate the sum of TF-IDF scores for each word
word_sums = tfidf_matrix.sum(axis=0)

# Create a Pandas Series of word sums using .A1
word_tfidf_scores = pd.Series(word_sums.A1, index=feature_names)

# Sort the Series in descending order
sorted_word_tfidf_scores = word_tfidf_scores.sort_values(ascending=False)

# Print the top 20 words
print("Top 20 most important terms in negative reviews based on TF-IDF:")
print(sorted_word_tfidf_scores.head(20))

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Check if the 'further_cleaned_review' column is empty
if negative_reviews_df['further_cleaned_review'].empty:
    print("The 'further_cleaned_review' column is empty. Cannot perform TF-IDF analysis.")
else:
    # Fit and transform the 'further_cleaned_review' column
    tfidf_matrix = tfidf_vectorizer.fit_transform(negative_reviews_df['further_cleaned_review'])

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Check if the 'further_cleaned_review' column is empty
if negative_reviews_df['further_cleaned_review'].empty:
    print("The 'further_cleaned_review' column is empty. Cannot perform TF-IDF analysis.")
else:
    # Fit and transform the 'further_cleaned_review' column
    tfidf_matrix = tfidf_vectorizer.fit_transform(negative_reviews_df['further_cleaned_review'])

   
    
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# Check if the tfidf_matrix is empty
  if tfidf_matrix.shape[0] == 0:
        print("The TF-IDF matrix is empty. Cannot perform TF-IDF analysis.")
    else:
        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

  # Calculate the sum of TF-IDF scores for each word
  word_sums = tfidf_matrix.sum(axis=0)

  # Convert to a 1D array and create a Pandas Series
  word_tfidf_scores = pd.Series(word_sums.A1, index=feature_names)

  # Sort the Series in descending order
  sorted_word_tfidf_scores = word_tfidf_scores.sort_values(ascending=False)

  # Print the top 20 words
  print("Top 20 most important terms in negative reviews based on TF-IDF:")
  print(sorted_word_tfidf_scores.head(20))


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Check if there are any non-empty reviews in the column
non_empty_reviews = negative_reviews_df['further_cleaned_review'][
    negative_reviews_df['further_cleaned_review'].str.strip().astype(bool)
]

if non_empty_reviews.empty:
    print("After cleaning and filtering, there are no non-empty reviews to analyze.")
else:
    # Fit and transform the non-empty reviews
    tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_reviews)

  # Check if the tfidf_matrix is empty
  if tfidf_matrix.shape[0] == 0:
        print("The TF-IDF matrix is empty after fitting and transforming.")
    else:
        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

  # Calculate the sum of TF-IDF scores for each word
  word_sums = tfidf_matrix.sum(axis=0)

  # ✅ FIXED: Convert sparse matrix to 1D array using .A1
  word_tfidf_scores = pd.Series(word_sums.A1, index=feature_names)

  # Sort the Series in descending order
  sorted_word_tfidf_scores = word_tfidf_scores.sort_values(ascending=False)

  # Print the top 20 words
  print("Top 20 most important terms in negative reviews based on TF-IDF:")
  print(sorted_word_tfidf_scores.head(20))


  from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Check if there are any non-empty reviews in the column
non_empty_reviews = negative_reviews_df['further_cleaned_review'][
    negative_reviews_df['further_cleaned_review'].str.strip().astype(bool)
]

if non_empty_reviews.empty:
    print("After cleaning and filtering, there are no non-empty reviews to analyze.")
else:
    # Fit and transform the non-empty reviews
    tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_reviews)

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Check if there are any non-empty reviews in the column
non_empty_reviews = negative_reviews_df['further_cleaned_review'][
    negative_reviews_df['further_cleaned_review'].str.strip().astype(bool)
]

if non_empty_reviews.empty:
    print("After cleaning and filtering, there are no non-empty reviews to analyze.")
else:
    # Fit and transform the non-empty reviews
    tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_reviews)

  # Check if the tfidf_matrix is empty
  if tfidf_matrix.shape[0] == 0:
        print("The TF-IDF matrix is empty after fitting and transforming.")
    else:
        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

  # Calculate the sum of TF-IDF scores for each word
  word_sums = tfidf_matrix.sum(axis=0)

  # ✅ Correct: convert sparse matrix to 1D array
  word_tfidf_scores = pd.Series(word_sums.A1, index=feature_names)

  # Sort the Series in descending order
  sorted_word_tfidf_scores = word_tfidf_scores.sort_values(ascending=False)

  # Print the top 20 words
  print("Top 20 most important terms in negative reviews based on TF-IDF:")
  print(sorted_word_tfidf_scores.head(20))

  from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Check if there are any non-empty reviews in the column
non_empty_reviews = negative_reviews_df['further_cleaned_review'][negative_reviews_df['further_cleaned_review'].str.strip().astype(bool)]

if non_empty_reviews.empty:
    print("After cleaning and filtering, there are no non-empty reviews to analyze.")
else:
    # Fit and transform the non-empty reviews
    tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_reviews)

  # Check if the tfidf_matrix is empty
  if tfidf_matrix.shape[0] == 0:
        print("The TF-IDF matrix is empty after fitting and transforming.")
    else:
        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

  # Calculate the sum of TF-IDF scores for each word and convert to dense array
  word_sums = tfidf_matrix.sum(axis=0).A1 # .A1 is a shortcut for toarray().flatten()

  # Create a Pandas Series of word sums
  word_tfidf_scores = pd.Series(word_sums, index=feature_names)

  # Sort the Series in descending order
  sorted_word_tfidf_scores = word_tfidf_scores.sort_values(ascending=False)

  # Print the top 20 words
  print("Top 20 most important terms in negative reviews based on TF-IDF:")
  print(sorted_word_tfidf_scores.head(20))

print("\nInterpretation of Top TF-IDF Terms in Negative Reviews:")
print("Based on the top 20 terms (sucks, insane, fake, idk, biased, etc.), the common issues driving negative reviews appear to be related to:")
print("- Performance or quality issues ('sucks', 'insane', 'fake')")
print("- Uncertainty or lack of knowledge ('idk')")
print("- Perceived bias ('biased')")
print("- Problems with access or functionality ('access', 'login', 'update', 'loading', 'servers', 'website', 'account', 'reach', 'sin', 'idk')")
print("- Content or output issues ('output', 'answers')")
print("\nSummary of Common Problems in Negative Reviews:")
print("The TF-IDF analysis of negative reviews indicates that users frequently complain about the performance and quality of the service, including issues with the accuracy or authenticity of the output. Additionally, many negative reviews highlight technical problems such as difficulties with access, logging in, updates, loading times, server issues, and account management. There is also a notable concern regarding perceived bias in the responses and general uncertainty or lack of helpfulness ('idk').")
# Convert 'Review Date' to datetime objects
df['Review Date'] = pd.to_datetime(df['Review Date'])

# Set 'Review Date' as the DataFrame index for time-series analysis
df.set_index('Review Date', inplace=True)

# Display the first few rows to verify
display(df.head())

# Aggregate sentiment by month (you can change 'M' to 'D' for daily, 'W' for weekly, etc.)
monthly_sentiment = df['compound_score'].resample('M').mean()

# Display the first few aggregated sentiment scores
display(monthly_sentiment.head())

# Visualize the monthly sentiment trend
plt.figure(figsize=(12, 6))
plt.plot(monthly_sentiment.index, monthly_sentiment.values)
plt.title('Monthly Average Compound Sentiment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Compound Sentiment Score')
plt.grid(True)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/50d62243-b9c3-431a-8ea7-24409c9d2276" />

# Calculate the value counts of the 'Ratings' column
rating_distribution = df['Ratings'].value_counts()

# Sort the value counts by the rating value
sorted_rating_distribution = rating_distribution.sort_index()

# Print the sorted value counts
print("Distribution of Ratings:")
print(sorted_rating_distribution)

# Group by 'Ratings' and calculate the mean sentiment scores
average_sentiment_by_rating = df.groupby('Ratings')[['compound_score', 'negative_score', 'neutral_score', 'positive_score']].mean()

# Print the resulting DataFrame
print("Average Sentiment Scores by Rating:")
print(average_sentiment_by_rating)

# Create a bar chart for the distribution of ratings
plt.figure(figsize=(8, 6))
plt.bar(sorted_rating_distribution.index, sorted_rating_distribution.values, color='skyblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(sorted_rating_distribution.index)
plt.show()

# Create box plots for sentiment scores by rating
sentiment_columns = ['compound_score', 'negative_score', 'neutral_score', 'positive_score']
for col in sentiment_columns:
    plt.figure(figsize=(10, 6))
    df.boxplot(column=[col], by='Ratings')
    plt.title(f'{col.replace("_", " ").title()} Distribution by Rating')
    plt.xlabel('Ratings')
    plt.ylabel(col.replace("_", " ").title())
    plt.suptitle('') # Suppress the default suptitle from boxplot
    plt.show()

  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/359337cf-15e1-4f35-a240-2917973c1785" />
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5d1be890-1fc8-4a66-b58b-f4e937ed5baf" />
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/13a1d51c-d14e-4ef4-be4b-cc4e6587a997" />
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/95f6f260-5187-49e8-abd5-d0617f0713fc" />
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/01f36b42-956e-45e4-af2b-ab56bbf217c0" />



  from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Filter for high and low rated reviews
high_ratings_df = df[df['Ratings'].isin([4, 5])].copy()
low_ratings_df = df[df['Ratings'].isin([1, 2])].copy()

# 2. Clean the text for both groups
high_ratings_df['cleaned_review_high'] = high_ratings_df['Review'].apply(clean_text)
low_ratings_df['cleaned_review_low'] = low_ratings_df['Review'].apply(clean_text)

# 3. Further clean by removing common and rare words
# Recalculate common and rare words based on the respective datasets
all_words_high = list(itertools.chain.from_iterable(high_ratings_df['cleaned_review_high'].str.split()))
word_counts_high = Counter(all_words_high)
common_words_high = {word for word, count in word_counts_high.items() if count / len(high_ratings_df) > common_threshold}
rare_words_high = {word for word, count in word_counts_high.items() if count < rare_threshold}

all_words_low = list(itertools.chain.from_iterable(low_ratings_df['cleaned_review_low'].str.split()))
word_counts_low = Counter(all_words_low)
common_threshold_low = 0.01 # Use same thresholds as before
rare_threshold_low = 5
common_words_low = {word for word, count in word_counts_low.items() if count / len(low_ratings_df) > common_threshold_low}
rare_words_low = {word for word, count in word_counts_low.items() if count < rare_threshold_low}


def remove_common_rare_high(text):
    return ' '.join([word for word in text.split() if word not in common_words_high and word not in rare_words_high])

def remove_common_rare_low(text):
    return ' '.join([word for word in text.split() if word not in common_words_low and word not in rare_words_low])


high_ratings_df['further_cleaned_review_high'] = high_ratings_df['cleaned_review_high'].apply(remove_common_rare_high)
low_ratings_df['further_cleaned_review_low'] = low_ratings_df['cleaned_review_low'].apply(remove_common_rare_low)


# 4. and 5. Apply TF-IDF
tfidf_vectorizer_high = TfidfVectorizer(max_df=0.95, min_df=2)
tfidf_vectorizer_low = TfidfVectorizer(max_df=0.95, min_df=2)

# Handle empty reviews before fitting TF-IDF
non_empty_reviews_high = high_ratings_df['further_cleaned_review_high'][high_ratings_df['further_cleaned_review_high'].str.strip().astype(bool)]
non_empty_reviews_low = low_ratings_df['further_cleaned_review_low'][low_ratings_df['further_cleaned_review_low'].str.strip().astype(bool)]


if non_empty_reviews_high.empty:
    print("No non-empty high-rated reviews to analyze.")
    sorted_word_tfidf_scores_high = pd.Series(dtype=float) # Create empty series
else:
    tfidf_matrix_high = tfidf_vectorizer_high.fit_transform(non_empty_reviews_high)
    feature_names_high = tfidf_vectorizer_high.get_feature_names_out()
    word_sums_high = tfidf_matrix_high.sum(axis=0).A1
    word_tfidf_scores_high = pd.Series(word_sums_high, index=feature_names_high)
    sorted_word_tfidf_scores_high = word_tfidf_scores_high.sort_values(ascending=False)


if non_empty_reviews_low.empty:
    print("No non-empty low-rated reviews to analyze.")
    sorted_word_tfidf_scores_low = pd.Series(dtype=float) # Create empty series
else:
    tfidf_matrix_low = tfidf_vectorizer_low.fit_transform(non_empty_reviews_low)
    feature_names_low = tfidf_vectorizer_low.get_feature_names_out()
    word_sums_low = tfidf_matrix_low.sum(axis=0).A1
    word_tfidf_scores_low = pd.Series(word_sums_low, index=feature_names_low)
    sorted_word_tfidf_scores_low = word_tfidf_scores_low.sort_values(ascending=False)


# 9. Print the top 20 words for each group
print("\nTop 20 most important terms in High-Rated Reviews (Ratings 4 and 5) based on TF-IDF:")
print(sorted_word_tfidf_scores_high.head(20))

print("\nTop 20 most important terms in Low-Rated Reviews (Ratings 1 and 2) based on TF-IDF:")
print(sorted_word_tfidf_scores_low.head(20))


print("\nInterpretation of Top TF-IDF Terms in High-Rated Reviews:")
print("Based on the top 20 terms ('cool', 'wonderful', 'fantastic', 'superb', 'fast', etc.), high-rated reviews frequently mention positive attributes and use strong positive adjectives. Specific positive aspects include performance ('fast', 'works'), usefulness for tasks like studying and homework ('study', 'homework', 'learning'), and general positive experiences ('friend', 'world', 'thing'). Terms like 'accurate' also suggest satisfaction with the quality of the output.")

print("\nInterpretation of Top TF-IDF Terms in Low-Rated Reviews:")
print("Based on the top 20 terms ('ok', 'occurred', 'poor', 'server', 'loading', 'hate', etc.), low-rated reviews highlight negative experiences and technical issues. Common complaints include problems with functionality ('occurred', 'server', 'loading', 'stopped', 'connection'), negative sentiment ('poor', 'hate'), and issues with generating or showing content ('generate', 'shows', 'pictures'). The term 'ok' appearing in low ratings might indicate a lukewarm or unsatisfactory experience despite not being overtly negative in all cases.")

print("\nSummary of Common Themes in High vs. Low Rated Reviews:")
print("High-rated reviews focus on the positive aspects of the service, emphasizing performance, usefulness for specific tasks (like studying), and overall satisfaction with the quality and experience. In contrast, low-rated reviews are dominated by technical problems (server issues, loading, connectivity), negative sentiment, and issues with the core functionality of generating or displaying content. The contrast in keywords clearly delineates the drivers of positive and negative user experiences.")


# Calculate the number of reviews per day
daily_reviews_count = df.resample('D').size()

# Print the head of the daily review count
print("Daily Number of Reviews:")
display(daily_reviews_count.head())

# Calculate the number of reviews per week
weekly_reviews_count = df.resample('W').size()

# Print the head of the weekly review count
print("\nWeekly Number of Reviews:")
display(weekly_reviews_count.head())

# Visualize the daily number of reviews over time
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(daily_reviews_count.index, daily_reviews_count.values)
ax1.set_title('Daily Number of Reviews Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Reviews')
plt.show()

# Visualize the weekly number of reviews over time
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(weekly_reviews_count.index, weekly_reviews_count.values)
ax2.set_title('Weekly Number of Reviews Over Time')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Reviews')
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/192422e1-8e2c-422f-89c5-e5d613ddf85e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/645b86c0-e974-4767-8dfd-a7986e0a793e" />

# Calculate the mean rating per week
weekly_average_rating = df['Ratings'].resample('W').mean()

# Print the head of the weekly average rating
print("Weekly Average Rating:")
display(weekly_average_rating.head())

# Visualize the weekly average rating over time
plt.figure(figsize=(12, 6))
plt.plot(weekly_average_rating.index, weekly_average_rating.values)
plt.title('Weekly Average Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/88753234-d922-40a9-8e55-3a5e46140b0b" />

# Calculate the rolling average of weekly review counts (4-week window)
weekly_reviews_rolling_avg = weekly_reviews_count.rolling(window=4).mean()

# Calculate the standard deviation of weekly review counts
weekly_reviews_std = weekly_reviews_count.std()

# Identify spike periods (weekly count > rolling average + 2 * standard deviation)
spike_threshold = weekly_reviews_rolling_avg + 2 * weekly_reviews_std
spike_periods = weekly_reviews_count[weekly_reviews_count > spike_threshold]

# Print the identified spike periods
print("\nPeriods with significant spikes in weekly review volume (Weekly Count > Rolling Avg + 2*Std Dev):")
print(spike_periods)

# Lower the spike detection threshold to 1.5 standard deviations above the rolling average
spike_threshold = weekly_reviews_rolling_avg + 1.5 * weekly_reviews_std
spike_periods = weekly_reviews_count[weekly_reviews_count > spike_threshold]

# Print the identified spike periods with the new threshold
print("\nPeriods with significant spikes in weekly review volume (Weekly Count > Rolling Avg + 1.5*Std Dev):")
print(spike_periods)


# 1. Filter the original DataFrame df to include only reviews within the identified spike periods.
# The spike_periods index contains the dates of these periods.
reviews_during_spikes = df[df.index.isin(spike_periods.index)].copy()

# 2. Calculate the descriptive statistics of the sentiment scores
sentiment_stats_during_spikes = reviews_during_spikes[[
    'compound_score',
    'negative_score',
    'neutral_score',
    'positive_score'
]].describe()

# 3. Print the descriptive statistics
print("Descriptive Statistics of Sentiment Scores During Spike Periods:")
print(sentiment_stats_during_spikes)

# 4. Compare the sentiment statistics during spike periods with the overall sentiment statistics
# Overall sentiment statistics were calculated in a previous step and stored in the output.
# Re-calculate overall stats for easy comparison
overall_sentiment_stats = df[[
    'compound_score',
    'negative_score',
    'neutral_score',
    'positive_score'
]].describe()

print("\nDescriptive Statistics of Sentiment Scores Overall:")
print(overall_sentiment_stats)

print("\nComparison:")
print("Sentiment during spike periods vs. Overall sentiment:")
print(sentiment_stats_during_spikes.loc['mean'] - overall_sentiment_stats.loc['mean'])


# Filter the original DataFrame df to include reviews within the spike weeks.
# Iterate through the identified spike periods (which are week end dates)
# and filter reviews that fall within the corresponding week.
reviews_during_spikes = pd.DataFrame()
for week_end_date in spike_periods.index:
    # Calculate the start date of the week (assuming week ends on Sunday, 'W-SUN')
    week_start_date = week_end_date - pd.Timedelta(days=6)
    # Filter reviews within this week range
    weekly_reviews = df[(df.index >= week_start_date) & (df.index <= week_end_date)].copy()
    # Append to the DataFrame
    reviews_during_spikes = pd.concat([reviews_during_spikes, weekly_reviews])

# Remove duplicate reviews if any (in case a review falls exactly on a week boundary and is included in two ranges, though unlikely with weekly intervals)
reviews_during_spikes = reviews_during_spikes[~reviews_during_spikes.index.duplicated(keep='first')].copy()


# Calculate the descriptive statistics of the sentiment scores for reviews during spike periods
sentiment_stats_during_spikes = reviews_during_spikes[[
    'compound_score',
    'negative_score',
    'neutral_score',
    'positive_score'
]].describe()

# Print the descriptive statistics for spike periods
print("Descriptive Statistics of Sentiment Scores During Spike Periods:")
display(sentiment_stats_during_spikes)

# Overall sentiment statistics were calculated in a previous step and stored in overall_sentiment_stats.
# Print overall sentiment statistics for comparison
print("\nDescriptive Statistics of Sentiment Scores Overall:")
display(overall_sentiment_stats)

# Compare the sentiment statistics during spike periods with the overall sentiment statistics
print("\nComparison (Mean Difference - Spike Periods vs. Overall):")
display(sentiment_stats_during_spikes.loc['mean'] - overall_sentiment_stats.loc['mean'])

# Define a function to categorize sentiment based on compound score
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to create a new 'sentiment_category' column
df['sentiment_category'] = df['compound_score'].apply(categorize_sentiment)

# Display the distribution of the new sentiment categories
print("Distribution of Sentiment Categories:")
display(df['sentiment_category'].value_counts())

# Display the first few rows with the new column
display(df[['Review', 'compound_score', 'sentiment_category']].head())

# Calculate the length of each review
df['review_length'] = df['Review'].astype(str).apply(len)

# Display the first few rows including the new 'review_length' column
display(df[['Review', 'review_length']].head())

# Print descriptive statistics for the 'review_length' column
print("Descriptive Statistics for Review Length:")
print(df['review_length'].describe())

# Create a histogram of the 'review_length' column
plt.figure(figsize=(10, 6))
plt.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/218f3dc8-f136-48b0-998f-7444b0ab0e4c" />


# Calculate the average review length for each rating level
average_length_by_rating = df.groupby('Ratings')['review_length'].mean()

# Print the average review length for each rating
print("Average Review Length by Rating:")
print(average_length_by_rating)

# Create a scatter plot of review length vs. compound sentiment score
plt.figure(figsize=(10, 6))
plt.scatter(df['review_length'], df['compound_score'], alpha=0.5)
plt.title('Review Length vs. Compound Sentiment Score')
plt.xlabel('Review Length')
plt.ylabel('Compound Score')
plt.show()

# Create box plots for review length by rating
plt.figure(figsize=(10, 6))
df.boxplot(column=['review_length'], by='Ratings')
plt.title('Review Length Distribution by Rating')
plt.xlabel('Ratings')
plt.ylabel('Review Length')
plt.suptitle('') # Suppress the default suptitle from boxplot
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4d98ee82-abb8-461c-9e3b-290c873fe4b3" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c2c2bc3a-d542-41fc-b0ee-4faa6306fc00" />

print("Summary of Review Length and Quality Analysis:")
print("\nKey Findings:")
print("- The average review length varies significantly across different rating levels. Lower ratings (1 and 2) tend to have longer reviews on average compared to higher ratings (4 and 5).")
print("- The scatter plot of review length versus compound sentiment score shows a general trend where shorter reviews are often associated with higher (more positive) compound scores, while longer reviews have a wider range of sentiment, including more negative scores.")
print("- The box plots clearly illustrate the distribution of review lengths for each rating. Reviews with 1 and 2 stars have a higher median length and a larger spread (indicating more variance in length) compared to 4 and 5-star reviews, which are predominantly shorter.")

print("\nInsights:")
print("- Users who leave lower ratings are more likely to write longer reviews, possibly to explain their negative experiences and provide detailed feedback on issues.")
print("- Conversely, users leaving high ratings tend to write shorter, more concise reviews, perhaps simply expressing satisfaction without extensive detail.")
print("- This suggests that review length can be an indicator of the user's emotional investment and the complexity of their feedback, with longer reviews often signaling specific problems or detailed opinions (both positive and negative, though more frequently negative in this dataset as seen in the average lengths).")
print("- Analyzing the content of longer reviews, especially those with lower ratings, could provide valuable insights into specific areas for product or service improvement.")


from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Define thresholds for short and long reviews
# Using quantiles to define short (e.g., below 25th percentile) and long (e.g., above 75th percentile)
short_threshold = df['review_length'].quantile(0.25)
long_threshold = df['review_length'].quantile(0.75)

# 2. Filter for short and long rated reviews
short_reviews_df = df[df['review_length'] <= short_threshold].copy()
long_reviews_df = df[df['review_length'] >= long_threshold].copy()

# 3. Clean the text for both groups (using the existing clean_text function)
short_reviews_df['cleaned_review_short'] = short_reviews_df['Review'].apply(clean_text)
long_reviews_df['cleaned_review_long'] = long_reviews_df['Review'].apply(clean_text)

# 4. Further clean by removing common and rare words
# Recalculate common and rare words based on the respective datasets
all_words_short = list(itertools.chain.from_iterable(short_reviews_df['cleaned_review_short'].str.split()))
word_counts_short = Counter(all_words_short)
# Using the same thresholds as before for consistency
common_words_short = {word for word, count in word_counts_short.items() if count / len(short_reviews_df) > common_threshold}
rare_words_short = {word for word, count in word_counts_short.items() if count < rare_threshold}


all_words_long = list(itertools.chain.from_iterable(long_reviews_df['cleaned_review_long'].str.split()))
word_counts_long = Counter(all_words_long)
common_words_long = {word for word, count in word_counts_long.items() if count / len(long_reviews_df) > common_threshold}
rare_words_long = {word for word, count in word_counts_long.items() if count < rare_threshold}


def remove_common_rare_short(text):
    return ' '.join([word for word in text.split() if word not in common_words_short and word not in rare_words_short])

def remove_common_rare_long(text):
    return ' '.join([word for word in text.split() if word not in common_words_long and word not in rare_words_long])


short_reviews_df['further_cleaned_review_short'] = short_reviews_df['cleaned_review_short'].apply(remove_common_rare_short)
long_reviews_df['further_cleaned_review_long'] = long_reviews_df['cleaned_review_long'].apply(remove_common_rare_long)


# 5., 6., 7., 8., 9. Apply TF-IDF and sort
tfidf_vectorizer_short = TfidfVectorizer(max_df=0.95, min_df=2)
tfidf_vectorizer_long = TfidfVectorizer(max_df=0.95, min_df=2)

# Handle empty reviews before fitting TF-IDF
non_empty_reviews_short = short_reviews_df['further_cleaned_review_short'][short_reviews_df['further_cleaned_review_short'].str.strip().astype(bool)]
non_empty_reviews_long = long_reviews_df['further_cleaned_review_long'][long_reviews_df['further_cleaned_review_long'].str.strip().astype(bool)]


if non_empty_reviews_short.empty:
    print("No non-empty short reviews to analyze.")
    sorted_word_tfidf_scores_short = pd.Series(dtype=float) # Create empty series
else:
    tfidf_matrix_short = tfidf_vectorizer_short.fit_transform(non_empty_reviews_short)
    feature_names_short = tfidf_vectorizer_short.get_feature_names_out()
    word_sums_short = tfidf_matrix_short.sum(axis=0).A1
    word_tfidf_scores_short = pd.Series(word_sums_short, index=feature_names_short)
    sorted_word_tfidf_scores_short = word_tfidf_scores_short.sort_values(ascending=False)


if non_empty_reviews_long.empty:
    print("No non-empty long reviews to analyze.")
    sorted_word_tfidf_scores_long = pd.Series(dtype=float) # Create empty series
else:
    tfidf_matrix_long = tfidf_vectorizer_long.fit_transform(non_empty_reviews_long)
    feature_names_long = tfidf_vectorizer_long.get_feature_names_out()
    word_sums_long = tfidf_matrix_long.sum(axis=0).A1
    word_tfidf_scores_long = pd.Series(word_sums_long, index=feature_names_long)
    sorted_word_tfidf_scores_long = word_tfidf_scores_long.sort_values(ascending=False)


# 10. Print the top 20 words for each group
print("\nTop 20 most important terms in Short Reviews (Length <= {}) based on TF-IDF:".format(short_threshold))
print(sorted_word_tfidf_scores_short.head(20))

print("\nTop 20 most important terms in Long Reviews (Length >= {}) based on TF-IDF:".format(long_threshold))
print(sorted_word_tfidf_scores_long.head(20))


print("\nInterpretation of Top TF-IDF Terms in Short Reviews:")
print("Based on the top 20 terms ('superb', 'helpful', 'op', 'fantastic', 'bad', etc.), short reviews tend to use concise, strong adjectives to describe their experience. They focus on overall impressions and basic functionality ('helpful', 'useful', 'work'). The presence of both positive ('superb', 'fantastic', 'wonderful') and negative ('bad') terms indicates that short reviews can express both positive and negative sentiment directly.")

print("\nInterpretation of Top TF-IDF Terms in Long Reviews:")
print("Based on the top 20 terms ('pictures', 'alot', 'full', 'easily', 'must', etc.), long reviews delve into more specific details and functionalities. Terms like 'pictures', 'platform', 'team', 'assignments', and 'create' suggest discussions about particular features, interactions with the service or support, and specific use cases. The presence of terms like 'wow', 'incredible', 'friendly', and 'appreciate' indicates that while longer reviews can detail issues, they also provide elaborate positive feedback.")

print("\nSummary of Common Themes in Short vs. Long Reviews:")
print("Short reviews provide quick, high-level feedback using strong sentiment words. They are useful for a general sense of user satisfaction or dissatisfaction. Long reviews, on the other hand, offer detailed feedback on specific features, performance, and user experience. They are valuable for identifying specific strengths and weaknesses of the service and understanding how it is used in practice.")



# Calculate Q1, Q3, and IQR for 'compound_score'
Q1_sentiment = df['compound_score'].quantile(0.25)
Q3_sentiment = df['compound_score'].quantile(0.75)
IQR_sentiment = Q3_sentiment - Q1_sentiment

# Define lower and upper bounds for sentiment score outliers
lower_bound_sentiment = Q1_sentiment - 1.5 * IQR_sentiment
upper_bound_sentiment = Q3_sentiment + 1.5 * IQR_sentiment

# Identify reviews with unusually high or low compound sentiment scores (outliers)
sentiment_outliers_df = df[(df['compound_score'] < lower_bound_sentiment) | (df['compound_score'] > upper_bound_sentiment)].copy()

# Display the number of identified sentiment score outliers
print(f"Number of sentiment score outliers: {len(sentiment_outliers_df)}")

# Display the first few rows of the DataFrame containing sentiment score outliers
print("\nFirst 5 rows of sentiment score outliers:")
display(sentiment_outliers_df.head())


# Calculate Q1, Q3, and IQR for 'review_length'
Q1_length = df['review_length'].quantile(0.25)
Q3_length = df['review_length'].quantile(0.75)
IQR_length = Q3_length - Q1_length

# Define lower and upper bounds for review length outliers
lower_bound_length = Q1_length - 1.5 * IQR_length
upper_bound_length = Q3_length + 1.5 * IQR_length

# Filter the DataFrame to identify reviews with unusually long or short lengths
review_length_outliers_df = df[(df['review_length'] < lower_bound_length) | (df['review_length'] > upper_bound_length)].copy()

# Print the number of identified review length outliers
print(f"Number of review length outliers: {len(review_length_outliers_df)}")

# Display the first 5 rows of the review_length_outliers_df DataFrame
print("\nFirst 5 rows of review length outliers:")
display(review_length_outliers_df.head())


# Calculate the rolling average of weekly review counts (4-week window)
weekly_reviews_rolling_avg = weekly_reviews_count.rolling(window=4).mean()

# Calculate the standard deviation of weekly review counts
weekly_reviews_std = weekly_reviews_count.std()

# Define spike threshold (rolling average + 1.5 * standard deviation)
spike_threshold = weekly_reviews_rolling_avg + 1.5 * weekly_reviews_std

# Identify periods where the weekly review count exceeds the spike threshold
spike_periods = weekly_reviews_count[weekly_reviews_count > spike_threshold]

# Print the identified spike periods
print("\nPeriods with significant spikes in weekly review volume (Weekly Count > Rolling Avg + 1.5*Std Dev):")
print(spike_periods)


from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Assume 'Review' column exists in your DataFrame
def get_sentiment_category(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply to all relevant DataFrames
for df in [review_length_outliers_df, sentiment_outliers_df, reviews_during_spikes]:
    if 'Review' in df.columns:
        df['sentiment_category'] = df['Review'].apply(get_sentiment_category)
# Sentiment distribution
print("Sentiment Distribution of Review Length Outliers:")
display(review_length_outliers_df['sentiment_category'].value_counts())
if 'sentiment_category' in review_length_outliers_df.columns:
    display(review_length_outliers_df['sentiment_category'].value_counts())
else:
    print("sentiment_category column not found.")

# Re-filter the original DataFrame df to include reviews within the spike weeks,
# ensuring the 'sentiment_category' column is included.
reviews_during_spikes = pd.DataFrame()
for week_end_date in spike_periods.index:
    # Calculate the start date of the week (assuming week ends on Sunday, 'W-SUN')
    week_start_date = week_end_date - pd.Timedelta(days=6)
    # Filter reviews within this week range
    weekly_reviews = df[(df.index >= week_start_date) & (df.index <= week_end_date)].copy()
    # Append to the DataFrame
    reviews_during_spikes = pd.concat([reviews_during_spikes, weekly_reviews])

# Remove duplicate reviews if any
reviews_during_spikes = reviews_during_spikes[~reviews_during_spikes.index.duplicated(keep='first')].copy()

# 6. Analyze the sentiment distribution of the reviews from the spike periods
print("\nSentiment Distribution of Reviews During Spike Periods:")
display(reviews_during_spikes['sentiment_category'].value_counts())

# 7. Analyze the rating distribution of the reviews from the spike periods
print("\nRating Distribution of Reviews During Spike Periods:")
display(reviews_during_spikes['Ratings'].value_counts().sort_index())

# 8. Display the first few rows of the reviews_during_spikes DataFrame
print("\nFirst 5 rows of reviews during spike periods:")
display(reviews_during_spikes.head())


















