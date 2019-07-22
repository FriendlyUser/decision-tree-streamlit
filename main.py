# parse craigs list
import pandas as pd
import os
import argparse
############### MACHINE LEARNING ###############################################
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus
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


def read_used_cars(csv_file='la_trimmed_features.csv'):
    """reads la used cars dataset
    """
    cars_df = pd.read_csv('la_cars_full.csv')
    X = cars_df.drop(["price"], axis=1)
    y = cars_df.price
    return X, y

#### MAIN logic here
def main():
    print("Doing something heere")
    if os.path.exists('la_cars_full.csv') == False:
        extract_data()

    X, y = read_used_cars()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = list(X.columns.values),class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')

if __name__ == "__main__":
    # This library handles argument parsing.  You don't need to worry about this for this assignment.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to image", default="pear.png") 

    # Add debug mode
    # parser.add_argument("-d","--debug", type=str2bool, nargs='?',
    #                     const=True, default=False,
    #                     help="Run Debug Functions")
    args = parser.parse_args()
    main()

