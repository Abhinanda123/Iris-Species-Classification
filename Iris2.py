# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating a Logistic Regression model. 
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

@st.cache()
def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
  species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor" 

st.sidebar.title("Iris Species Prediction App")

# Add 4 sliders and store the value returned by them in 4 separate variables. 
s_len = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
# The 'float()' function converts the 'numpy.float' values to Python float values.
s_wid = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
p_len = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
p_wid = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

classifier= st.sidebar.selectbox("Classifier",("Random Forest Classifier", "Logistic Regression", "Support Vector Machines"))

if st.sidebar.button("Predict"):

	if classifier == "Random Forest Classfier":
		pred= prediction(rf_clf, s_len, s_wid, p_len, p_wid)
		acc= rf_clf.score(X_train, y_train)

	elif classifier == "Logistic Regression":
		pred= prediction(log_reg, s_len, s_wid, p_len, p_wid)
		acc= log_reg.score(X_train, y_train)

	else:
		pred= prediction(svc_model, s_len, s_wid, p_len, p_wid)
		acc= svc_model.score(X_train, y_train)

	st.write("Species name: ", pred)
	st.write("Accuracy of chosen classifier: ", acc)

	if pred=="Iris-setosa":
		st.image("https://i.pinimg.com/originals/49/38/3a/49383aabb5d7eb54e8c747ee18dc2f47.jpg")
	elif pred== "Iris-virginica":
		st.image("https://www.plant-world-seeds.com/images/item_images/000/003/886/large_square/IRIS_VIRGINICA.JPG?1495391090")
	else:
		st.image("https://cdn.shopify.com/s/files/1/0739/9053/products/Blue-iris-1_1024x1024.jpg?v=1478538058")

