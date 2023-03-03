#!/usr/bin/env python
# coding: utf-8

# # Travel Insurance Claim Prediction

# __Context__
# - Travel insurance is a type of insurance that covers the costs and losses associated with traveling. It is useful protection for those traveling domestically or abroad.
# - Many companies selling tickets or travel packages, give consumers the option to purchase travel insurance, also known as travelers insurance. Some travel policies cover damage to personal property, rented equipment, such as rental cars, or even the cost of paying a ransom.
# 
# __Problem Statement__
# - As a data scientist in an insurance company in the USA. The company has collected the data of earlier travel insurance buyers. In this season of vacation, the company wants to know which person will claim their travel insurance and who will not.
# 
# __Objective__
# - You are responsible for building a machine learning model for the insurance company to predict if the insurance buyer will claim their travel insurance or not.
# 
# __Evaluation Criteria__
# - Submissions are evaluated using F1 Score.

# ##### About the data
# There are 11 columns in the dataset. Some of them are mentioned below:
# 
# __Duration:__ Travel duration
# 
# __Destination:__ Travel destination
# 
# __Agency:__ Agency Name
# 
# __Commission:__ Commission on the insurance
# 
# __Age:__ Age of the insurance buyer
# 
# __Gender:__ Gender of the insurance buyer
# 
# __Agency Type:__ What is the agency type?
# 
# __Distribution Channel:__ offline/online
# 
# __Product Name:__ Name of the insurance plan
# 
# __Net Sales:__ Net sales
# 
# __Claim:__ If the insurance is claimed or not (the target variable), 0 = not claimed, 1 = claimed

# In[179]:


#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
#from sklearn.ensemble
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
#from sklearn.metrics import 

get_ipython().system('pip install pywedge')
import pywedge as pw
import six

import warnings
warnings.filterwarnings(action='ignore')


# In[180]:


insurance_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/travel_insurance/Training_set_label.csv" )

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/travel_insurance/Testing_set_label.csv')
insurance_data.head()

#test_data.head()


# In[181]:


df = insurance_data.copy()
test_df = test_data.copy()
df.head(3)
#test_df.head(3)


# In[182]:


df.tail(3)


# In[183]:


df.shape , test_df.shape


# In[184]:


df.info()


# __Observations__
# - Theres a total of total 11 columns and  48260 entries
# - The Duration ,Age ,and the target Claim  variable are of the int 64 data types.The  Net Sales,Commision (in value) are of the float types.
# - The remaining columns are object datatypes
# -  There are  48260 non-null values in 10 of the colums with missing values in Gender column,i.e ,__Ther eare missing values in gender column__

# ### Perform Exploratory Data Analysis¶

# In[ ]:





# In[185]:


(df.isnull().sum() /df.shape[0]) * 100


# In[186]:


df.drop(columns = ['Gender'],inplace = True)
df.head(3)


# In[187]:


df.duplicated().sum()


# In[188]:


df.drop_duplicates(inplace=True)
df.shape ,test_df.shape


# In[ ]:





# In[189]:


#checking the value counts of the categorical
for i in df.select_dtypes(include=object):
    
    print(df[i].value_counts(ascending=False))
    #print(df[i].value_counts(ascending=False , normalize = True))
    print("-" * 40)


# In[190]:


df.describe().T


# In[197]:


df_claim = df[insurance_data['Claim'] == 1]
df_claim.shape[0]
#df_claim['Gender'].value_counts()


# Now we will check the string type columns one by one,But first we need to check if the value found are the same in both sets.

# In[198]:


print(len(test_data.Agency.value_counts()), len(df.Agency.value_counts()))
test_data.Agency.value_counts().index.isin((insurance_data.Agency.value_counts().index))


# In[199]:


print(len(test_data['Product Name'].value_counts()),len(df['Product Name'].value_counts()))

test_data['Product Name'].value_counts().index.isin(df['Product Name'].value_counts().index)


# In[200]:


print(len(test_data['Agency Type'].value_counts()), len(df['Agency Type'].value_counts()))


test_data['Agency Type'].value_counts().index.isin(df['Agency Type'].value_counts().index)


# In[201]:


print(len(test_data['Distribution Channel'].value_counts()), len(df['Distribution Channel'].value_counts()))

test_data['Distribution Channel'].value_counts().index.isin(df['Distribution Channel'].value_counts().index)


# In[202]:


print(len(test_data['Destination'].value_counts()), len(df['Destination'].value_counts()))
test_data['Destination'].value_counts().index.isin(insurance_data['Destination'].value_counts().index)


# We have found some unmatched values so we're going to further investigate it

# In[203]:


destinations = df['Destination'].value_counts().index

test_destinations_unmatched = test_data[~test_data['Destination'].isin(destinations)]
test_destinations_unmatched.Destination.value_counts()


# All of them only appear once so we'll look into it when we address this column

# In[204]:


#we will look at how many different values there are in each column, and how much each one appears
len(df.Agency.value_counts()), df.Agency.value_counts()


# In[205]:


df['Agency Type'].value_counts()


# In[206]:


df['Distribution Channel'].value_counts()


# In[207]:


len(df['Product Name'].value_counts()), df['Product Name'].value_counts()


# In[208]:


#let's check the average number of times a product appears, and the median of its appearance
df['Product Name'].value_counts().mean(), df['Product Name'].value_counts().median()


# In[209]:


len(df['Destination'].value_counts()), df['Destination'].value_counts()


# In[210]:


#let's check the average number of times a destination appears, and the median of its appearance
df['Destination'].value_counts().mean(), df['Destination'].value_counts().median()


# In[211]:


#creating a list with all the destination that have less than 10 appearances
destination_frequency = df['Destination'].value_counts()
destination_sub10 = destination_frequency .loc[destination_frequency <10]

#list of destinations of countries with less than 10 appearances
destination_sub10.index


# In[212]:


#total number of appearances of destinations that appear 10 or less times vs destinations that appear more than 10
destination_over10 = destination_frequency .loc[destination_frequency >=10]
destination_over10.sum(), destination_sub10.sum()


# - We will encode the string like columns,
# - However, since the destination column has a lot of values with low -frequency(<10, 226 appearances total, vs 41754 of the remaining destinations) we will transform all this destinations into a new category called others and afterwards we will apply the dummification.

# In[213]:


#getting the list with condition above and replacing it in the original dataframe
others_destination = destination_sub10.index
df.Destination.replace(others_destination,'OTHERS',inplace=True)
#checking the new frequency
df.Destination.value_counts()


# In[214]:


#checking the new list of destinations
df.Destination.value_counts().index


# In[215]:


df.columns


# In[216]:



'''
INPUT:
df - pandas dataframe with categorical variables you want to dummy
cat_cols - list of strings that are associated with names of the categorical columns
dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

OUTPUT:
df - a new dataframe that has the following characteristics:
1. contains all columns that were not specified as categorical
2. removes all the original columns in cat_cols
3. dummy columns for each of the categorical columns in cat_cols
4. if dummy_na is True - it also contains dummy columns for the NaN values
5. Use a prefix of the column name with an underscore (_) for separating
'''
def create_dummy_df(dfr, cat_cols, dummy_na):
    
    for col in cat_cols:
        try:
            dfr= pd.concat([dfr.drop(columns=col, axis=1), pd.get_dummies(dfr[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)# for each cat add dummy var, drop original column
        except:
            
            continue  
    return dfr;

dummy_na = False
dummy_cols =['Agency', 'Agency Type', 'Distribution Channel', 'Product Name','Destination']

#dummifyin the columns above
df = create_dummy_df(df,dummy_cols,dummy_na)
df


# Now we we will take a look at the continuos/discrete type columns, by quickly plotting its distribution

# In[217]:


test_data.columns


# In[218]:


cont_cols =['Duration', 'Net Sales', 'Commision (in value)', 'Age']
cont_cols


# In[219]:


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = [ax for axes_row in axes for ax in axes_row]
for i, c in enumerate(df[cont_cols]):
    
    plot = sns.distplot(df[c] ,ax=axes[i])
plt.tight_layout()


# __Observations__
# - The duration plot suggest there are outliers in this column, so we will further investigate this
# - The variable appears to all be rightly skewed

# In[220]:


#plot the boxplot to visualize outliers
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = [ax for axes_row in axes for ax in axes_row]
for i, c in enumerate(df[cont_cols]):
    
    plot = sns.boxplot(df[c] ,ax=axes[i])
plt.tight_layout()


# In[221]:


df['Duration'].value_counts()


# In[222]:


#let's see how many values are above 1000 days
df.loc[df.loc[:,'Duration']>1000]


# In[223]:


#let's see how the distrubtion looks like without the values above
df.loc[df.loc[:,'Duration']<1000,'Duration'].plot.hist(bins=10)


# In[224]:


#we can observe that the majority of the values are under 100, so let's look at the distribution below 100
df.loc[df.loc[:,'Duration']<100,'Duration'].plot.hist(bins=10)


# Looking at this distribution an initial assumption would be to categorize the Destination column in the following way: steps of 20 till 100, then steps of 100 till 500, and a final category above 500

# In[225]:


#applying the categorization described above

bins = [0,20,40,60,80,100,200,300,400,500, np.inf]
names = ["<20",'20-40', '40-60', '60-80', '80-100', '100-200', '200-300', '300-400', '400-500','500+']

df['Duration'] = pd.cut(df['Duration'], bins, labels=names)
df


# In[226]:


#doing the same analysis for the Net Sales column
df['Net Sales'].plot.hist(bins=10)


# In[227]:


#we can see that the majority of the values are around 0 so let's take a closer look at that window
df.loc[(df.loc[:,'Net Sales']<100) & (df.loc[:,'Net Sales']>-100) ,'Net Sales'].plot.hist(bins=10)


# In[228]:


#looking at the 0-50 window
df.loc[(insurance_data.loc[:,'Net Sales']<50) & (df.loc[:,'Net Sales']>0) ,'Net Sales'].plot.hist(bins=10)


# In[229]:


#0-100 window
df.loc[(df.loc[:,'Net Sales']<100) & (df.loc[:,'Net Sales']>0) ,'Net Sales'].plot.hist(bins=10)


# In[230]:


#checking the maximum and minimum values
df['Net Sales'].max(),df['Net Sales'].min()


# So according to these values a good starting point might be: <-200, -200 to -100, -100 to -50, -50 to -25, -25 to 0, then steps of 5 to 50, then steps of 10 to 100, then steps of 100 to 400, and finally 400+

# In[231]:


#applying the categorization described above
bins_commission = [-np.inf, 1, 5, 10, 20, 30, 40, 50, 100, 150, np.inf]
names_commission = ['<1', '1-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100-150','150+']

df['Commision (in value)'] = pd.cut(df['Commision (in value)'], bins_commission, labels=names_commission)
df


# In[232]:


df['Age'].plot.hist(bins=10)


# In[233]:


#at a first glance the frequency above 100 years seems very, let's take a further look at it
df.loc[df.loc[:,'Age']>100]


# In[234]:


df.loc[df.loc[:,'Age']>100].Age.value_counts()


# __Observations__
# - Two things strike as very odd, one is that there are 441 people over 100 years travelling, 
# - and second and even most odd that all of these have exactly 118 years old. So my first hypothesis is that there is an inputation error in these rows. So before moving on, we'll go on the safe side and remove these lines.

# In[235]:


df = df.loc[df.loc[:,'Age']<100]
df


# In[236]:


#taking a new look after removing the above va
df['Age'].plot.hist(bins=10)


# In[237]:


df['Age'].plot.hist(bins=20)


# The approach this time will be:
# 
# <20, then steps of 5 till 60, then steps of 10 till 80 and finally 80+

# In[238]:


#applying the categorization described above
bins_age = [-np.inf,20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, np.inf]
names_age = ['<20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60','60-70','70-80',np.inf]

df['Age'] = pd.cut(df['Age'], bins_age, labels=names_age)
df


# In[239]:


#now we can apply the above function to dummify the newly categorized columns
dummy_cols =['Duration', 'Net Sales', 'Commision (in value)', 'Age']

#dummifying the columns above
df = create_dummy_df(df,dummy_cols,dummy_na)
df


# In[240]:


#finally let's take a look at the distribution of the target variable
df.Claim.value_counts()


# __Observations__
# - This is a very unbalanced dataset so we will need to apply a balancing technique and we'll go with smoteen

# In[241]:


from sklearn.model_selection import train_test_split

#but first we'll need to split the target variable and the data into training and test sets

y = df['Claim']
X = df.drop(['Claim'], axis = 1)

X.shape , y.shape

# We'll split into 80% training, 20% test, and finally set the random_state = 42 to keep reproductability
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
train_X.shape ,test_X.shape, train_y.shape, test_y.shape


# In[242]:


train_X.head()


# In[243]:


from imblearn.over_sampling import SMOTE
#Synthetic Minority Oversampling Technique

sms = SMOTE(random_state= 25,sampling_strategy = 1.0)#We are equalizing both the classes

X_train , y_train = sms.fit_resample(train_X ,train_y)
X_train.head()


# 
# from imblearn.combine import SMOTEENN
# 
# 
# #balancing the data
# sm = SMOTEENN(random_state=42 )
# train_X_smote, train_y_smote = sm.fit_resample(train_X, train_y)
# 

# The next step will be to build a pipeline to not only tune the model parameters but to test more than one model at once.
# -  an estimator is an equation for picking the “best,” or most likely accurate, data model based upon observations
# 
#  make_pipeline generates names for steps automatically.

# #to do so we must import the base estimator class
# from sklearn.base import BaseEstimator
# from xgboost import XGBClassifier
# 
# class ClfSwitcher(BaseEstimator):
#     def __init__(self, estimator= XGBClassifier(),):
#         
#         
#         self.estimator = estimator
# 
#     def fit(self, X, y=None, **kwargs):
#         
#         self.estimator.fit(X, y)
#         
#         return self
#     
# 
#     def predict(self, X, y=None):
#         return self.estimator.predict(X)
#     
#     def predict_proba(self, X):
#         return self.estimator.predict_proba(X)
# 
#     def score(self, X, y):
#         return self.estimator.score(X, y)
#     
# 

# from sklearn.pipeline import Pipeline
# 
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# 
# from sklearn.metrics import make_scorer,accuracy_score,recall_score,precision_score ,f1_score , classification_report,confusion_matrix
# from sklearn.metrics import matthews_corrcoef
# 
# 
# #since we have seen this is a very unbalanced dataset, even after performing oversampling to balance the data,
# #a good way to evaluate the model is to focus on the prediction on the less represented class,
# #for that a good evaluation metric will be the recall of the class 1, because it tells how good that class is being predicted.
# #On top of that we will add the Matthews correlation coeficient that gives an overall score of how the model is performing.
# 
# scoring = {
# 'Recall': make_scorer(recall_score, pos_label=1,average='binary'),
# 'MCC':make_scorer(matthews_corrcoef),"F1": make_scorer(f1_score) }
# 
# def model_pipeline():
#     
#     
#     pipeline = Pipeline([
#     ('clf', ClfSwitcher())
#      ])
#     
#     # specify parameters for grid search
#     parameters = [
#         {
# 'clf__estimator': [RandomForestClassifier()],
# 'clf__estimator__n_estimators': [5, 25, 50, 75, 100, 150, 200, 250],
# 'clf__estimator__max_features': [0.4, 0.6, 0.8, 1 ],
# 'clf__estimator__max_depth': [3, 4, 5, 6, 7],
# 'clf__estimator__random_state': [42],
# 
# },
# 
# 
# ]
#     
#  
#      #create grid search object
#     cv = GridSearchCV(Pipeline,
#     param_grid=parameters,
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     scoring=scoring,
#     refit='Recall',
#     return_train_score=True)
#     return cv
# 
# #return cv

# #we will now instatiate the pipeline and initiate the grid search
# cv = model_pipeline()
# cv.fit(X_train, y_train)

# from imblearn.over_sampling import BorderlineSMOTE
# exp_name = setup(data = df, target = 'Claim', use_gpu=True, ignore_low_variance = True, session_id=111,fix_imbalance = True,fix_imbalance_method=BorderlineSMOTE(sampling_strategy=1) )
# compare_models(sort = 'F1')

#  ruamel-yaml pip install ruamel-yaml

# from pycaret.classification import *
# from imblearn.over_sampling import BorderlineSMOTE
# exp_name = setup(data = df, target = 'Claim')# train_size = 0.8, data_split_shuffle=True, session_id = 2)
# compare_models(sort = 'F1')

# lr = create_model('lr')
# rf = create_model('rf')
# 

# from sklearn.linear_model import LogisticRegression
# scoring = 'accuracy'
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('QDA', QuadraticDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# models.append(('RFC', RandomForestClassifier(n_estimators=10)))
# models.append(('ADA', AdaBoostClassifier(n_estimators=100)))
# models.append(('MLPC', MLPClassifier(solver='lbfgs', alpha=1e5,hidden_layer_sizes=(5, 5), random_state=1)))
# results=[]
# names=[]
# for name, model in models:
#     
#     kfold=model_selection.KFold(n_splits=10,random_state=seed)
#     cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring="accuracy")
#     results.append(cv_results)
#     names.append(name)
#     mpdels.set.shuffle=True
#     msg="%s:%f" % (name,cv_results.mean())
#     print(msg)

# In[295]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np

def model_check(models, X_train, y_train):
    
    for name, model in models.items():
        
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)
    print(f'{name} F1 score : {np.mean(score)}')

models = {'random_forest':RandomForestClassifier(),
'logistic_reg':LogisticRegression(),
'XGB':XGBClassifier(),
'GB':GradientBoostingClassifier()}

print("Various Models")
model_check(models, X_train, y_train)
print()


# In[299]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

model = RandomForestClassifier(n_jobs=-1, verbose=1)
param_dist = {'n_estimators':[300, 400, 500, 600], 'max_depth':[5,6,7,8]}

random = RandomizedSearchCV(model, param_dist, random_state=0, scoring='f1', n_jobs=-1, cv=3, verbose=1)
search = random.fit(X_train, y_train)
print('BEST PARAM', search.best_params_)


# In[297]:


pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score')


# In[298]:


from sklearn.metrics import f1_score
model = RandomForestClassifier(n_estimators=400, max_depth=8)
score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

print(f'Model F1 score : {np.mean(score)}')


# In[300]:


model.fit(X_train, y_train)


# In[302]:


X_train.shape ,X_test.shape


# In[ ]:


X_test.isnull().sum()


# In[ ]:


test_data.drop(['Claim'], axis=1, inplace=True)


# In[ ]:


predict = model.predict(X_test)


# In[ ]:


predict


# In[ ]:


res = pd.DataFrame(predict) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = test_data.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]


# In[ ]:


res["prediction"] = res["prediction"].astype("object")


# In[ ]:


res=res.reset_index()
res=res.drop(['index'],axis=1)


# In[ ]:


res


# In[ ]:


res.to_csv('prediction_results.csv')

