import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree  
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title('Iris Classification using Decision Trees')


# Viewing the dataframe
st.markdown("#### View of the dataset")
df = pd.read_csv('Iris.csv')
st.dataframe(df)


# View the null values for each column
null_values, quick_summaries = st.columns((3, 3))

with null_values:
    st.markdown('#### View the null values for each column')
    nulls = df.isna().sum()
    st.dataframe(nulls)
  
    
with quick_summaries:
    st.markdown('### Quick Summaries')
    rows, columns = df.shape
    classes = df['Species'].unique()
    st.write("Number of columns:", columns)
    st.write("Number of Rows", rows)
    st.write("Number of classes:", len(classes))
    st.write(classes)

st.markdown('# Now for the scikit part....')


target, features = st.columns((0.5, 3))


le = LabelEncoder()
df['labels'] = le.fit_transform(df["Species"])
X = np.array(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])                   # Input
Y = np.array(df[["labels"]])   


with target:
    st.dataframe(df['labels'])

with features:
    st.dataframe(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 100)                                    # 80:20

st.markdown('## Decision Tree')                                                                        

classifier_type = st.selectbox('Select Criterion', ['gini', 'entropy'])

def dtc_parameter(selectbox):
    params = dict()
    if selectbox == 'Gini':
        splitter = st.select_slider('Splitter', ['Best', 'Random'])
        max_depth = st.slider('Depth of tree', 0, 100, 0)
        min_samples_split = st.slider('Min # of samples', 0, 20, 2)
        min_samples_leaf = st.slider('Min # of samples', 0, 20, 1)
        
        params['Split Type'] = splitter
        params['Max Depth'] = max_depth
        params['Split Samples'] = min_samples_split
        params['Leaf Samples'] = min_samples_leaf
    else:
        splitter = st.select_slider('Splitter for Entropy', ['best', 'random'])
        max_depth = st.slider('Depth of tree', 0, 100, 0)
        min_samples_split = st.slider('Min # of samples', 0, 20, 2)
        min_samples_leaf = st.slider('Min # of samples', 0, 20, 1)
        
        params['Split Type'] = splitter
        params['Max Depth'] = max_depth
        params['Split Samples'] = min_samples_split
        params['Leaf Samples'] = min_samples_leaf
    
    return params

paramsdtc = dtc_parameter(classifier_type)


result = st.selectbox('Whatchu want boi', ['Train Model', 'Visualize Model'])

# clf_gini = DecisionTreeClassifier(criterion = classifier_type,
#                                   splitter = params['Split Type']
#                                   params['Max Depth'] = 
#                                   params['Split Samples'] = min_samples_split
#                                   params['Leaf Samples'] = min_samples_leaf

def dtc_actions(selectbox, params, classifier_type, X_train, y_train, X_test):
    if selectbox == 'Train Model':
        clf = DecisionTreeClassifier(criterion = classifier_type,
                                     splitter = params['Split Type'],
                                     max_depth = params['Max Depth'],
                                     min_samples_split = params['Split Samples'],
                                     min_samples_leaf = params['Leaf Samples'],
                                     )
      
        clf.fit(X_train, y_train)
        
        dtc_pred = clf.predict(X_test)
        
        return st.dataframe(dtc_pred)
        
    else:
        clf = DecisionTreeClassifier(criterion = classifier_type,
                                     splitter = params['Split Type'],
                                     max_depth = params['Max Depth'],
                                     min_samples_split = params['Split Samples'],
                                     min_samples_leaf = params['Leaf Samples'],
                                     )
      
        clf.fit(X_train, y_train)
        
        fig = plt.figure(figsize=(25,20))
        fff = tree.plot_tree(clf, 
                   filled=True)
        
        plot = tree.plot_tree(clf)
        
        return st.pyplot(fig)

dtc_final = dtc_actions(result, paramsdtc, classifier_type, 
                        X_train, Y_train, X_test)






# def get_classifier(clf_name, params):
#     clf = None
#     if clf_name == 'SVM':
#         clf = SVC(C=params['C'])
#     elif clf_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     else:
#         clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
#             max_depth=params['max_depth'], random_state=1234)
#     return clf




# classifier_name = st.sidebar.selectbox('Select classifier', ('Decision Trees'))

# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == 'SVM':
#         C = st.sidebar.slider('C', 0.01, 10.0)
#         params['C'] = C
#     elif clf_name == 'KNN':
#         K = st.sidebar.slider('K', 1, 15)
#         params['K'] = K
#     else:
#         max_depth = st.sidebar.slider('max_depth', 2, 15)
#         params['max_depth'] = max_depth
#         n_estimators = st.sidebar.slider('n_estimators', 1, 100)
#         params['n_estimators'] = n_estimators
#     return params

# params = add_parameter_ui(classifier_name)