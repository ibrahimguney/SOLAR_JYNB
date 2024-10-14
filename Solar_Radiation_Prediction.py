#!/usr/bin/env python
# coding: utf-8

# # Solar Radiation Prediction
# 
# **Data Description**
# 
# The actual solar radiation under a specific time was determined from laboaratory. Data is in raw form(not scaled). The data has 10 quantitative input variables, and 1 quantitaive output variable and 32686 instances (observations).
# 
# **Domain**
# 
# Envirnomental Aspects
# 
# **Context**
# 
# Radiation is the most important topic for envinoment as well as for human beings, The solar radiation prediction is a highly non linear function of age and ingredients. These ingredients include UNIX time, Data, Time, Temperature, Pressure, Humidity, WindDirection(Degrees), Speed, Time Sun Rise and Time Sun Set.
# 
# **Attribute Information**
# 
# * UNIXTime : Unix time is currently defined as the number of seconds which have passed since 00:00:00 UTC on Thursday, 1 January 1970, which is referred to as the Unix epoch.
# * Data : measured in yyyy-mm-dd format 
# * Time : measured in hh:mm:ss 24-hour format
# * Radiation : measured in watts per meter^2
# * Temperature : measured in degrees Fahrenheit
# * Pressure : measured in Hg
# * Humidity : measured in percent
# * WindDirection(Degrees) : measured in degrees
# * Speed : measured in miles per hour
# * TimeSunRise/TimeSunSet : Hawaii time
# 
# **Objective**
# 
# Modeling of solar radiation of high performance prediction using Machine Learning

# __________________________________________________________________________________________________________________________________________________________________________________________________

# **importing the libraries**

# In[1]:


import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , FunctionTransformer , OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# **importing the data**

# In[2]:


df = pd.read_csv(r"C:\Users\emran\Downloads\SolarPrediction.csv")
df.head()


# **Summary of data**

# In[3]:


df.shape


# In[4]:


df.info()


# # Univariate Analysis

# In[5]:


df.describe()


# In[6]:


fig = px.box(df, x="Radiation")
fig.show()


# In[7]:


fig = px.box(df, x="Temperature")
fig.show()


# In[8]:


fig = px.box(df, x="Pressure")
fig.show()


# In[9]:


fig = px.box(df, x="Humidity")
fig.show()


# In[10]:


fig = px.box(df, x="Speed")
fig.show()


# In[11]:


fig = px.box(df, x="WindDirection(Degrees)")
fig.show()


# *Conclusions that can be made out of it:*
# 
# * all the features given in the data are numeric expect last two is string
# * There are no NULL or NaN values present in the data
# * In total there are 10 features and 32686 observations
# * We shall be using radiation as the dependent variable and all other features as independent variable as it makes the best sense out of the problem objective.
# * from the plotly we can plotted and individual attribute's boxplots which were plotted before are the evidences that there are outliers present in the attributes: Radiation, Temperature, Pressure, Speed and WindDirection(Degrees)

# ________________________________________________________________________________________________________________________________________________________________________________

# # Multivariate Analysis

# In[12]:


sns.pairplot(df)
plt.show()


# In[13]:


df.hist(bins=25, figsize=(12,8))
plt.show()


# *Conclusion that can be drawn from bi-variate analysis:*
# 
# * Pressure, Humidity and WindDirection(Degrees) are the attributes which are rightly skewed.
# * Temperature and Speed are the attributes which are rightly skewed.

# # Split the Data

# In[14]:


df['Temperature_cat'] = pd.cut(df['Temperature'], bins=[30, 35, 40, 45, 50, 55, 60, 65, 70, np.inf], 
                               labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])
df.head()


# In[15]:


from sklearn.model_selection import train_test_split
strat_train, strat_test = train_test_split(df, test_size=0.3, stratify=df["Temperature_cat"], random_state=42)


# In[16]:


strat_train.drop("Temperature_cat", axis=1, inplace=True)
strat_test.drop("Temperature_cat", axis=1, inplace=True)


# In[17]:


strat_train


# In[18]:


strat_test


# *Conclusion that can be drawn from split data:*
# 
# * Temperature attribute is more useful or important attribute to predict the solar radiation.
# * training dataset should be representative of data.
# * So we used stratified sampling and also split the data into 70% for training data and 30% for testing data.

# # Relationship between dependent and independent variables

# In[19]:


df = strat_train.copy()


# In[20]:


df[["Radiation", "Temperature", "Pressure", "Humidity", "WindDirection(Degrees)", "Speed"]].corr()


# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(df[["Radiation", "Temperature", "Pressure", "Humidity", "WindDirection(Degrees)", "Speed"]].corr(), 
            annot= True, cmap='BuPu', linecolor='black')
plt.title("Correlation between the variables")
plt.show()


# *Conclusion that can be drawn from relationship between dependent and independent variables:*
# 
# * Radiation vs Temperature: is highly positively correlated, it's also linearly related
# * Radiation vs Pressure: is positively correalted with less degree of correaltion
# * Radiation vs Humidity: is negatively correlated of almost -0.23
# * Radiation vs WindDirection(Degrees): is negatively correlated of almost -0.24
# * Radiation vs Speed: positive and fairly correalted of almost 0.069

# **Make each attributes into Gausian distributions**

# In[22]:


sy = PowerTransformer("yeo-johnson")
df1 = sy.fit_transform(df[["Temperature", "Pressure", "Humidity", "WindDirection(Degrees)", "Speed"]])


# In[23]:


sy.lambdas_


# **Check how much outliers is in the training set**

# In[24]:


df2 = df[(df["Radiation"]>883.66)|(df["Radiation"]>883.66)|(df["Temperature"]>68) | (df["Pressure"]<30.31) | (df["Speed"]>14.62) | 
         (df["WindDirection(Degrees)"]>325.49)]


# In[25]:


df2.shape


# * There total 3549 outliers in the training set.
# * But we don't remove this outliers because it is right information and also it will use full for predictions.

# ____________________________________________________________________________________________________________________________________________________________________________________

# # Creating pipeline for data preprocessing

# In[26]:


df = strat_train.drop("Radiation", axis=1)
df_labels = strat_train["Radiation"].copy()


# In[27]:


num_pipeline = make_pipeline(SimpleImputer(strategy="median"), PowerTransformer("yeo-johnson"), StandardScaler())
preprocessing = ColumnTransformer([("num", num_pipeline, ["Temperature", "Pressure", "Humidity", "WindDirection(Degrees)",
                                                        "Speed"]),])


# In[28]:


df_prepared = preprocessing.fit_transform(df)
df_prepared.shape


# In[29]:


preprocessing.get_feature_names_out()


# *In this pipeline we used imputation if there is filling value, converting attributes into gauisan distribution and scaling variables.*

# # Select and train model

# **Algorithm**:- *Linear Regression*

# In[30]:


from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(df, df_labels)


# In[31]:


lin_predictions = lin_reg.predict(df)
lin_predictions[:5].round(2)


# In[32]:


df_labels.iloc[:5]


# In[33]:


lin_rmse = mean_squared_error(df_labels, lin_predictions, squared=False)
print("RMSE: ", lin_rmse)


# In[34]:


from sklearn import metrics
lin_reg_r2 = metrics.r2_score(df_labels, lin_predictions)
print("R^2 Score:", lin_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(df_labels, lin_predictions))


# **Algorithm**:- *Decision Tree Regressor*

# In[35]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor())
tree_reg.fit(df, df_labels)


# In[36]:


tree_predictions = tree_reg.predict(df)
tree_rmse = mean_squared_error(df_labels, tree_predictions)
tree_rmse


# *RMSE is 1.47 but training data overfit the model so we used validation set i.e cross validation method for evaluating the model.*

# In[37]:


tree_rmses = -cross_val_score(tree_reg, df, df_labels, cv=10, scoring="neg_root_mean_squared_error")
pd.Series(tree_rmses).describe()


# In[38]:


tree_reg_r2 = metrics.r2_score(df_labels, tree_predictions)
print("R^2 Score:", tree_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(df_labels, tree_predictions))


# **Algorithm**:- *Support Vector Regressor*

# In[39]:


from sklearn.svm import SVR

svr_reg = make_pipeline(preprocessing, SVR())
svr_reg.fit(df, df_labels)


# In[40]:


svr_rmses = -cross_val_score(tree_reg, df, df_labels, cv=10, scoring="neg_root_mean_squared_error")
pd.Series(svr_rmses).describe()


# In[41]:


svr_predictions = svr_reg.predict(df)


# In[42]:


svr_reg_r2 = metrics.r2_score(df_labels, svr_predictions)
print("R^2 Score:", svr_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(df_labels, svr_predictions))


# **Algorithm**:- *Random Forest Regressor*

# In[43]:


from sklearn.ensemble import RandomForestRegressor

rnd_reg = make_pipeline(preprocessing, RandomForestRegressor())
rnd_reg.fit(df, df_labels)


# In[44]:


rnd_rmses = -cross_val_score(rnd_reg, df, df_labels, cv=10, scoring="neg_root_mean_squared_error")
pd.Series(rnd_rmses).describe()


# In[45]:


rnd_predictions = rnd_reg.predict(df)
rnd_rmse=mean_squared_error(df_labels, rnd_predictions, squared=False)
rnd_rmse


# In[46]:


rnd_reg_r2 = metrics.r2_score(df_labels, rnd_predictions)
print("R^2 Score:", rnd_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(df_labels, rnd_predictions))


# In[47]:


rnd_predictions[:5].round(2)


# *Conclusion that can be drawn from model selection:*
# 
# * Random Forest Regressor is the best model for this because mean of rmses is very low compare to other.
# * So we will hyperparameter tune to random forest regressor.

# **Hyper Parameter Tuning of Random Forest Regressor**

# In[48]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import Pipeline

full_pipeline = Pipeline([("preprocessing", preprocessing), ("random_forest", RandomForestRegressor(random_state=42))])

param_distribs = {'random_forest__max_features': randint(low=0, high=6), 
                  'random_forest__n_estimators': randint(low=0, high=200),
                 'random_forest__max_depth': randint(low=0, high=100),
                 'random_forest__bootstrap': [True, False]}

rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, 
                               scoring="neg_root_mean_squared_error")
rnd_search.fit(df, df_labels)


# In[49]:


rnd_search.best_params_


# In[50]:


-rnd_search.best_score_


# *We used randomizedsearchcv because it help to find the best parameters for fixed range.*

# _____________________________________________________________________________________________________________________________________________________________________________________

# **Final model**

# In[51]:


final_model = rnd_search.best_estimator_
final_model


# In[52]:


feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)


# In[53]:


sorted(zip(feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True)


# # Evaluate in test set

# In[54]:


X_test = strat_test.drop("Radiation", axis=1)
y_test = strat_test["Radiation"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)


# In[55]:


final_model_r2 = metrics.r2_score(y_test, final_predictions)
print("R^2 Score:", final_model_r2)


# In[57]:


print(f"Accuracy of Hyperparameter tuned Random Forest Regressor is {100*final_model_r2}%.")


# *We can conclude that RMSE is low for test data so it show that it is generalized and R squared is 0.74 so we can say that 74% is the accuracy of our hyperparameter tune Random Forest Regressor model.*

# In[58]:


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# *You get a fairly large interval from 156 to 165, and your previous point estimate of 160 is roughly in the middle of it.*
# 
# *If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation. That’s because your system ends up fine-tuned to perform well on the validation data and will likely not perform as well on unknown datasets. That’s not the case in this example since the test RMSE is lower than the validation RMSE, but when it happens you must resist the temptation to tweak the hyperparameters to make the numbers look good on the test set; the
# improvements would be unlikely to generalize to new data.*
