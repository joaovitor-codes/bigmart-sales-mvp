import pandas as pd

def read_csv_smart(file_or_path):
    return pd.read_csv(file_or_path, sep=None, engine="python")
