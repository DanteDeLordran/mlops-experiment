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
    return


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
def _(plt):
    # Create figure and axes
    plt.figure(figsize=(10,6))
    return


if __name__ == "__main__":
    app.run()
