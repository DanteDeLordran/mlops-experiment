import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("TfIdf Max Features")

    df = pd.read_csv("./data/preproccessed_data.csv").dropna(subset=['clean_comment'])

    return (
        RandomForestClassifier,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        confusion_matrix,
        df,
        mlflow,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(
    RandomForestClassifier,
    TfidfVectorizer,
    accuracy_score,
    classification_report,
    confusion_matrix,
    df,
    mlflow,
    plt,
    sns,
    train_test_split,
):
    def run_experiment_tfidf_max_features(max_features):
        ngram_range = (1, 3)  # Trigram setting

        # Step 2: Vectorization using TF-IDF with varying max_features
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

        X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Step 4: Define and train a Random Forest model
        with mlflow.start_run() as run:
            # Set tags for the experiment and run
            mlflow.set_tag("mlflow.runName", f"TFIDF_Trigrams_max_features_{max_features}")
            mlflow.set_tag("experiment_type", "feature_engineering")
            mlflow.set_tag("model_type", "RandomForestClassifier")

            # Add a description
            mlflow.set_tag("description", f"RandomForest with TF-IDF Trigrams, max_features={max_features}")

            # Log vectorizer parameters
            mlflow.log_param("vectorizer_type", "TF-IDF")
            mlflow.log_param("ngram_range", ngram_range)
            mlflow.log_param("vectorizer_max_features", max_features)

            # Log Random Forest parameters
            n_estimators = 200
            max_depth = 15

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Initialize and train the model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Step 5: Make predictions and log metrics
            y_pred = model.predict(X_test)

            # Log accuracy
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            # Log classification report
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics in classification_rep.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric}", value)

            # Log confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix: TF-IDF Trigrams, max_features={max_features}")
            plt.savefig("./results/confusion_matrix_tfidf.png")
            mlflow.log_artifact("./results/confusion_matrix_tfidf.png")
            plt.close()

            # Log the model
            mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_trigrams_{max_features}")

    # Step 6: Test various max_features values
    max_features_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    for max_features in max_features_values:
        run_experiment_tfidf_max_features(max_features)
    return


if __name__ == "__main__":
    app.run()
