import pandas as pd
import requests


def load_raw(path):
    df = pd.read_csv(path)
    print(df.columns)
    print(df.count)
    return df
