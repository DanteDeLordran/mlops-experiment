import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    df = pd.read_csv("./data/data.csv")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df = df[~(df["clean_comment"].str.strip() == "")]

    nltk.download('stopwords')
    nltk.download('wordnet')

    def preprocess_comment(comment: str):
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    return df, preprocess_comment


@app.cell
def _(df, preprocess_comment):
    df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
    df.head()
    return


@app.cell
def _(df):
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    vectorizer = CountVectorizer(max_features=10000)

    x = vectorizer.fit_transform(df['clean_comment']).toarray()
    y = df['category']

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("RF Baseline")
    return (
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mlflow,
        plt,
        sns,
        train_test_split,
        vectorizer,
        x,
        y,
    )


@app.cell
def _(
    RandomForestClassifier,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mlflow,
    plt,
    sns,
    train_test_split,
    vectorizer,
    x,
    y,
):
    # Split data into training and tests sets (20% test, 80% training)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)

    # Define and train a random forest baseline model using a simple train-test split
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", "RandomForest_Baseline_TrainTestSplit")
        mlflow.set_tag("experiment_type", "baseline")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        mlflow.set_tag("description", "Baseline RandomForest model for sentiment analysis using Bag of Words with")

        mlflow.log_param("vectorizer_type", "CountVectorizer")
        mlflow.log_param("vectorize_max_features", vectorizer.max_features)

        n_estimators = 200
        max_depth = 15

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f'{label}_{metric}', value)

        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix")

        plt.savefig("/results/confusion_matrix.png")
        mlflow.log_artifact("/content/confusion_matrix.png")

    print(f'Accuracy: {accuracy}')
    return


@app.cell
def _(df):
    df.to_csv('preproccessed_data.csv', index=False)

    return


if __name__ == "__main__":
    app.run()
