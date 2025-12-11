import pandas as pd
def load_sample(path='data/synthetic_hiring_small.csv'):
    return pd.read_csv(path)