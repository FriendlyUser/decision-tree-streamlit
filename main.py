# parse craigs list
import pandas as pd
import os
import argparse
############### MACHINE LEARNING ###############################################
from sklearn.tree import DecisionTreeRegressor # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
import pydotplus
from sklearn.model_selection import KFold
import numpy as np
def extract_data():
    """Extract data of interest from the Used cars dataset from kaggle
    https://www.kaggle.com/austinreese/craigslist-carstrucksdata/downloads/craigslist-carstrucks-data.zip/7 . 
    """
    cars_df = pd.read_csv('craigslist-carstrucks-data/craigslistVehiclesFull.csv')

    # Get list of unique city names, city of interest is losangeles
    # print(cars_df.city.unique())
    cars_df = cars_df[cars_df['city'].str.contains("Los|Angeles|Los Angeles|los")]

    # Part A save
    cars_df.to_csv('la_cars_full.csv')

    # Part B save
    cars_df = cars_df.drop(["url", "city", "image_url", "state_code", "state_name", "county_name"], axis=1)
    cars_df.to_csv('la_trimmed_features.csv')
    # https://www.datacamp.com/community/tutorials/decision-tree-classification-python
    # features to remove include url, city, image_url, state_code, state_name, county_name, lat, long
    # url, city, price, year, manufacturer, make, condition, cylinders, fuel, odometer, title_status, transmission, vin, drive, size, type, paint_color, image_url, lat, long, county_fips, county_name, state_fips, state_code, state_name, weather


# part c
def read_used_cars(csv_file='la_trimmed_features.csv'):
    """reads la used cars dataset
    """
    cars_df = pd.read_csv(csv_file)
    copy_df = cars_df.drop(["price"], axis=1)
    copy_df = cars_df.fillna(value=0)
    # numerical_features = copy_df.dtypes == 'float'
    numerical_features = copy_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = copy_df.select_dtypes(exclude=["number","bool_","object_"]).columns.tolist()
    preprocess = make_column_transformer(
        (StandardScaler(), numerical_features),
        (OneHotEncoder(), categorical_features))
    X = preprocess.fit_transform(copy_df)
    y = cars_df["price"]
    return X, y, numerical_features.append(categorical_features)

def main():
    print("Doing something heere")
    if os.path.exists('la_cars_full.csv') == False:
        extract_data()
    # part d
    X, y, feature_list = read_used_cars()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test
    # Create Decision Tree classifer object
    clf = DecisionTreeRegressor(max_depth=10)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = KFold(n_splits=10)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Mean Error:",metrics.mean_absolute_error(y_test, y_pred))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_list,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('sample_graph.png')
    # Train decision tree with kfold cross validation
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    fold = 0
    global_df = pd.DataFrame()
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Model Accuracy, how often is the classifier correct?
        print("MSE:",metrics.mean_absolute_error(y_test, y_pred))
        simple_list = [fold, 
            metrics.mean_absolute_error(y_test, y_pred), 
            metrics.r2_score(y_test, y_pred),
            metrics.median_absolute_error(y_test, y_pred)]
        df=pd.DataFrame([simple_list],columns=['k','Mean Absolute error', 'r2 score', 'Median Absolute Error'])
        global_df = global_df.append(df)
        fold = fold + 1
    global_df.to_latex('testing.tex')


if __name__ == "__main__":
    main()

