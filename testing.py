import pandas as pd
def read_used_cars(csv_file='la_trimmed_features.csv'):
    """reads la used cars dataset
    """
    cars_df = pd.read_csv('la_trimmed_features.csv')
    X = cars_df.drop(["price"], axis=1)
    y = cars_df.price
    return X, y


X, y = read_used_cars()