#!/usr/bin/env python
# coding: utf-8

# ## Food recommendation system

# ### Define the objective in business terms: A food recommendation system is a tool that is designed to help businesses in the food industry improve customer satisfaction and loyalty by providing personalized recommendations for food and drinks based on an individual's preferences and past orders. The primary objective of a food recommendation system is to increase sales and revenue by encouraging customers to try new menu items and by encouraging repeat business. This can be achieved by analyzing customer data and using machine learning algorithms to understand patterns in customer behavior and preferences, and by presenting recommendations that are tailored to the individual's tastes and needs. By using a food recommendation system, businesses can improve their customer retention rates, increase customer satisfaction, and ultimately drive growth and profitability.
# 
# ### How will the solution be used: The solution will be used in a website application or even a mobile app.
# 
# ### How should you frame this problem ( supervised/unsupervised, online/offline,...): The problem is an unsupervised,offline and a model based learning problem, since we are building a recommendation model from the whole unlabled data.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('recipes.csv')


# In[3]:


df.head(5)


# ## Exploring the data

# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.tail(5)


# In[8]:


df.nunique()


# In[9]:


df.isnull().sum()


# In[11]:


(df.isnull().sum()/len(df))*100


# ## Data cleaning

# In[12]:


df.isnull().sum()


# In[16]:


copy_df = df.copy()


# In[17]:


copy_df.RecipeServings = pd.to_numeric(copy_df.RecipeServings, errors='coerce')


# In[18]:


copy_df.info()


# In[20]:


copy_df.isnull().sum()


# In[22]:


copy_df.loc[copy_df["RecipeServings"].isnull() == True]


# In[23]:


(copy_df.isnull().sum())*100/copy_df.shape[0]


# In[24]:


copy_df.dropna(how = "any",inplace = True)


# In[26]:


copy_df.isnull().sum()


# In[28]:


ana_copy_df = df.copy() #for later analysis


# In[30]:


# Group the tenure in bins of 12 months
labels = ["{0}-{1}".format(i, i + 11) for i in range(1, 72, 12)]

copy_df['Calories'] = pd.cut(df.Calories, range(1, 80, 12), right=False, labels=labels)

copy_df.drop(columns='Calories', axis=1, inplace=True)
copy_df.head()


# In[31]:


ana_copy_df.describe()


# In[48]:


ana_copy_df.describe(include="all").T


# In[49]:


#seperatingcategorical and numerical features before analysis
cat_cols=ana_copy_df.select_dtypes(include=['object']).columns
num_cols = ana_copy_df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)


# In[51]:


fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Frequency Histogram')
plt.ylabel('Frequency')
plt.xlabel('Bins Center')
ax.hist(df.Calories.to_numpy(),bins=[0,100,200,300,400,500,600,700,800,900,1000,1000,2000,3000,5000],linewidth=0.5, edgecolor="white")
plt.show()


# In[53]:


import pylab 
import scipy.stats as stats
stats.probplot(df.Calories.to_numpy(), dist="norm", plot=pylab)
pylab.show()


# ### We start by extracting the columns that we are interessted in. Since we are building a recommendation engine that takes the recipes nutritional characteristics, we start by extracting a sub data with the relevant columns. We may still need other columns for our project. However, we will mainly use the columns with nutritional information for trainning our model.

# In[55]:


dataset = df.copy()
columns=['RecipeId','Name','CookTime','PrepTime','TotalTime','RecipeIngredientParts','Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent','RecipeInstructions']
dataset = dataset[columns]


# In[60]:


max_Calories=2000
max_daily_fat=100
max_daily_Saturatedfat=13
max_daily_Cholesterol=300
max_daily_Sodium=2300
max_daily_Carbohydrate=325
max_daily_Fiber=40
max_daily_Sugar=40
max_daily_Protein=200
max_list=[max_Calories,max_daily_fat,max_daily_Saturatedfat,max_daily_Cholesterol,max_daily_Sodium,max_daily_Carbohydrate,max_daily_Fiber,max_daily_Sugar,max_daily_Protein]


# In[62]:


extracted_data = dataset.copy()
for column,maximum in zip(extracted_data.columns[6:15],max_list):
    extracted_data = extracted_data[extracted_data[column]<maximum]


# In[63]:


extracted_data.info()


# In[64]:


extracted_data.iloc[:,6:15].corr()


# In[65]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
prep_data=scaler.fit_transform(extracted_data.iloc[:,6:15].to_numpy())


# In[66]:


prep_data


# ## Training the model

# In[68]:


from sklearn.neighbors import NearestNeighbors

# Create a NearestNeighbors instance with cosine metric
neigh = NearestNeighbors(metric="cosine", algorithm="brute")

# Fit the model to the prepared data
neigh.fit(prep_data)


# In[72]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(neigh.kneighbors,kw_args={'return_distance':False})
pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])


# In[74]:


params={'n_neighbors':10,'return_distance':False}
pipeline.get_params()
pipeline.set_params(NN__kw_args=params)


# In[75]:


pipeline.transform(extracted_data.iloc[0:1,6:15].to_numpy())[0]


# ## Testing the model

# In[76]:


extracted_data.iloc[pipeline.transform(extracted_data.iloc[0:1,6:15].to_numpy())[0]]


# ## K nearest Neighbour

# In[78]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


df.sample(3)


# In[79]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a k-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# ## Random forest classifier
# 

# In[80]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# ## Decision Tree classifier

# In[83]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)
# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# ## Gaussian

# In[84]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


# In[85]:


extracted_data[extracted_data['RecipeIngredientParts'].str.contains("egg",regex=False)]


# ## Creating an end to end function

# In[86]:


def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

def extract_data(dataframe,ingredient_filter,max_nutritional_values):
    extracted_data=dataframe.copy()
    for column,maximum in zip(extracted_data.columns[6:15],max_nutritional_values):
        extracted_data=extracted_data[extracted_data[column]<maximum]
    if ingredient_filter!=None:
        for ingredient in ingredient_filter:
            extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient,regex=False)] 
    return extracted_data
def apply_pipeline(pipeline,_input,extracted_data):
    return extracted_data.iloc[pipeline.transform(_input)[0]]

def recommand(dataframe,_input,max_nutritional_values,ingredient_filter=None,params={'return_distance':False}):
    extracted_data=extract_data(dataframe,ingredient_filter,max_nutritional_values)
    prep_data,scaler=scaling(extracted_data)
    neigh=nn_predictor(prep_data)
    pipeline=build_pipeline(neigh,scaler,params)
    return apply_pipeline(pipeline,_input,extracted_data)


# In[87]:


test_input=extracted_data.iloc[0:1,6:15].to_numpy()
recommand(dataset,test_input,max_list)


# In[88]:


dataset.to_csv('dataset.csv',index=False)
dataset.to_json('dataset.json')


# In[89]:


import matplotlib.pyplot as plt

# Algorithm names
algorithms = ['KNN', 'DECISION TREE','RANDOM FOREST','GAUSSIAN NB','ANN']

# Corresponding accuracy values
accuracy = [94.50,83.50,92.00,86.50,81.38]

# Create a bar graph
plt.barh(algorithms, accuracy, color='purple')
# Add labels and title
plt.xlabel('Algorithms')
plt.ylabel('Accuracy (%)')
plt.title('Algorithm Accuracy Comparison')
for i in range(len(algorithms)):
    plt.text(accuracy[i] + 1, i, f'{accuracy[i]}%', ha='left', va='center')


# Show the plot
plt.show()


# In[ ]:




