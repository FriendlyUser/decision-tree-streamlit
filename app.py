import streamlit as st
import pandas as pd
import numpy as np

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

st.title('Decision Tree ECE 470')

st.markdown("Basic Decision Tree Created for an academic class and I decided to test out streamlit with it.")

@st.cache
def load_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/FriendlyUser/decision-tree-streamlit/master/la_trimmed_features.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text('Loading data... done!')

@st.cache
def read_used_cars():
    """reads la used cars dataset
    """
    copy_df = data.drop(["price"], axis=1)
    copy_df = data.fillna(value=0)
    # numerical_features = copy_df.dtypes == 'float'
    numerical_features = copy_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = copy_df.select_dtypes(exclude=["number","bool_","object_"]).columns.tolist()
    preprocess = make_column_transformer(
        (StandardScaler(), numerical_features),
        (OneHotEncoder(), categorical_features))
    X = preprocess.fit_transform(copy_df)
    y = data["price"]
    return X, y, numerical_features.append(categorical_features)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Car Data')
hist_values = np.histogram(data["price"], bins=24, range=(0,18000))[0]
st.bar_chart(hist_values)

# TODO DEPLOY TO HEROKU, show decision tree image, or not give background on why I made this app
# Add Integer slider for max_depth, keep accuracy there
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
st.write("Mean Error: ", metrics.mean_absolute_error(y_test, y_pred))

# Plot Price vs brand
# Plot Price vs year
# Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
