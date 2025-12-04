import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("./data/data.csv")
    df.head()
    df.shape


if __name__ == "__main__":
    main()
