import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load CSV data
    """)
    return


@app.cell
def _(pd):
    # Load and preprocess the data in one cell
    df = pd.read_csv("./data/data.csv")

    # Drop nulls
    df = df.dropna()

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove empty comments (after stripping whitespace)
    df = df[~(df["clean_comment"].str.strip() == "")]

    # Convert to lowercase
    df["clean_comment"] = df["clean_comment"].str.lower()

    # Strip whitespace
    df["clean_comment"] = df["clean_comment"].str.strip()

    # Remove newlines
    df["clean_comment"] = df["clean_comment"].str.replace("\n", "", regex=True)
    return (df,)


@app.cell
def _(df):
    # Display basic info
    df.head()
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.sample()["clean_comment"].values
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    # Check for any remaining nulls (should be 0)
    df.isnull().sum()
    return


@app.cell
def _(df):
    # Check for any remaining duplicates (should be 0)
    df.duplicated().sum()
    return


@app.cell
def _(df):
    # Check for trailing/leading whitespaces (should be 0)
    df["clean_comment"].apply(lambda x: x.endswith(" ") or x.startswith(" ")).sum()
    return


@app.cell
def _(df):
    # Identify comments containing URLs
    import re

    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    comments_with_urls = df[df["clean_comment"].str.contains(url_pattern, regex=True)]
    # Display the comments containing URLs
    comments_with_urls.head()
    return (re,)


@app.cell
def _(df):
    # Check for any remaining newlines (should be empty)
    comments_with_newline = df[df["clean_comment"].str.contains("\n")]
    comments_with_newline
    return


@app.cell
def _(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.countplot(data=df, x="category")
    plt.show()
    return plt, sns


@app.cell
def _(df):
    # Frequency distribution of sentiments
    df["category"].value_counts(normalize=True).mul(100).round(2)
    return


@app.cell
def _(df):
    df['word_count'] = df['clean_comment'].apply(lambda x: len(x.split()))

    df['word_count'].describe()
    return


@app.cell
def _(df, sns):
    sns.displot(df['word_count'], kde=True)
    return


@app.cell
def _(df, plt, sns):
    # Create figure and axes
    plt.figure(figsize=(10,6))

    sns.kdeplot(df[df['category'] == 1]['word_count'], label='Positive', fill=True)

    sns.kdeplot(df[df['category'] == 0]['word_count'], label='Neutral', fill=True)

    sns.kdeplot(df[df['category'] == -1]['word_count'], label='Negative', fill=True)

    plt.title('Word count distribution by category')
    plt.xlabel('World count')
    plt.ylabel('Density')

    plt.legend()

    plt.show()
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10,6))

    sns.scatterplot(data=df, x='category', y='word_count', alpha=0.5)

    plt.title('Scatterplot of word count by category')
    plt.xlabel('Category')
    plt.ylabel('Word count')

    plt.show()
    return


@app.cell
def _(df):
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    df['num_stop_words'] = df['clean_comment'].apply(lambda x: len([word for word in x.split() if word in stop_words]))

    df.sample(5)
    return nltk, stop_words, stopwords


@app.cell
def _(df, plt, sns):
    # Create a distribution plot for the 'num_stop_words' column

    plt.figure(figsize=(10,6))
    sns.histplot(df['num_stop_words'], kde=True)
    plt.title('Distribution of Stop Word Count in Comments')
    plt.xlabel('Number of Stop Words')
    plt.ylabel('Frequency')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10,6))

    sns.kdeplot(df[df['category'] == 1]['num_stop_words'], label='Positive', fill=True)
    sns.kdeplot(df[df['category'] == 0]['num_stop_words'], label='Neutral', fill=True)
    sns.kdeplot(df[df['category'] == -1]['num_stop_words'], label='Negative', fill=True)

    plt.title('Num stop words Distribution by Category')
    plt.xlabel('Stop word count')
    plt.ylabel('Density')

    plt.legend()
    plt.show()
    return


@app.cell
def _(df, sns):
    sns.barplot(df,x='category', y='num_stop_words', estimator='median')
    return


@app.cell
def _(df, pd, plt, sns, stop_words):
    from collections import Counter

    all_stop_words = [word for comment in df['clean_comment'] for word in comment.split() if word in stop_words]

    most_common_stop_words = Counter(all_stop_words).most_common(25)

    top_25_df = pd.DataFrame(most_common_stop_words, columns=['stop_word', 'count'])

    plt.figure(figsize=(12,8))
    sns.barplot(data=top_25_df, x='count', y='stop_word', palette='viridis')
    plt.title('Top 25 Most Common Stop Words')
    plt.xlabel('Count')
    plt.ylabel('Stop Word')
    plt.show()
    return (Counter,)


@app.cell
def _(df):
    df['num_chars'] = df['clean_comment'].apply(len)
    df['num_chars'].describe()
    return


@app.cell
def _(df):
    df['num_punctuation_chars'] = df['clean_comment'].apply(
        lambda x: sum([1 for char in x if char in '.,!?;:"()[]{}-_'])
    )
    df['num_punctuation_chars'].describe()
    return


@app.cell
def _(df, pd, plt, sns):
    from sklearn.feature_extraction.text import CountVectorizer

    def get_top_ngrams(corpus, n=None):
        vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    top_25_bigrams = get_top_ngrams(df['clean_comment'], 25)

    top_25_bigrams_df = pd.DataFrame(top_25_bigrams, columns=['bigram', 'count'])

    plt.figure(figsize=(12,8))
    sns.barplot(data=top_25_bigrams_df, x='count', y='bigram', palette='magma')
    plt.title('Top 25 Most Common Bigrams')
    plt.xlabel('Count')
    plt.ylabel('Bigram')
    plt.show()
    return


@app.cell
def _(Counter, df, pd, re):


    df['clean_comment'] = df['clean_comment'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x)))

    all_text = ' '.join(df['clean_comment'])

    # Count the frequency of each character
    char_frequency = Counter(all_text)

    # Convert the character frequency into a DataFrame for better display
    char_frequency_df = pd.DataFrame(char_frequency.items(), columns=['character', 'frequency']).sort_values(by='frequency', ascending=False)

    char_frequency_df
    return


@app.cell
def _(df, stopwords):

    # Defining stop words but keeping essential ones for sentiment analysis
    new_stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

    # Remove stop words from 'clean_comment' column, retaining essential ones
    df['clean_comment'] = df['clean_comment'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in new_stop_words])
    )
    return


@app.cell
def _(df, nltk):
    from nltk.stem import WordNetLemmatizer

    nltk.download('wordnet')

    # Define the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Apply lemmatization to the 'clean_comment_no_stopwords' column
    df['clean_comment'] = df['clean_comment'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
    )

    df.head()
    return


@app.cell
def _(Counter, df, plt, sns):
    def plot_top_n_words(df, n=20):
        """Plot the top N most frequent words in the dataset."""
        # Flatten all words in the content column
        words = ' '.join(df['clean_comment']).split()

        # Get the top N most common words
        counter = Counter(words)
        most_common_words = counter.most_common(n)

        # Split the words and their counts for plotting
        words, counts = zip(*most_common_words)

        # Plot the top N words
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f'Top {n} Most Frequent Words')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.show()

    # Example usage
    plot_top_n_words(df, n=50)

    return


@app.cell
def _(df, plt):
    def plot_top_n_words_by_category(df, n=20, start=0):
        """Plot the top N most frequent words in the dataset with stacked hue based on sentiment category."""
        # Flatten all words in the content column and count their occurrences by category
        word_category_counts = {}

        for idx, row in df.iterrows():
            words = row['clean_comment'].split()
            category = row['category']  # Assuming 'category' column exists for -1, 0, 1 labels

            for word in words:
                if word not in word_category_counts:
                    word_category_counts[word] = { -1: 0, 0: 0, 1: 0 }  # Initialize counts for each sentiment category

                # Increment the count for the corresponding sentiment category
                word_category_counts[word][category] += 1

        # Get total counts across all categories for each word
        total_word_counts = {word: sum(counts.values()) for word, counts in word_category_counts.items()}

        # Get the top N most frequent words across all categories
        most_common_words = sorted(total_word_counts.items(), key=lambda x: x[1], reverse=True)[start:start+n]
        top_words = [word for word, _ in most_common_words]

        # Prepare data for plotting
        word_labels = top_words
        negative_counts = [word_category_counts[word][-1] for word in top_words]
        neutral_counts = [word_category_counts[word][0] for word in top_words]
        positive_counts = [word_category_counts[word][1] for word in top_words]

        # Plot the stacked bar chart
        plt.figure(figsize=(12, 8))
        bar_width = 0.75

        # Plot negative, neutral, and positive counts in a stacked manner
        plt.barh(word_labels, negative_counts, color='red', label='Negative (-1)', height=bar_width)
        plt.barh(word_labels, neutral_counts, left=negative_counts, color='gray', label='Neutral (0)', height=bar_width)
        plt.barh(word_labels, positive_counts, left=[i+j for i,j in zip(negative_counts, neutral_counts)], color='green', label='Positive (1)', height=bar_width)

        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.title(f'Top {n} Most Frequent Words with Stacked Sentiment Categories')
        plt.legend(title='Sentiment', loc='lower right')
        plt.gca().invert_yaxis()  # Invert y-axis to show the highest frequency at the top
        plt.show()



    plot_top_n_words_by_category(df, n=20)
    return


if __name__ == "__main__":
    app.run()
