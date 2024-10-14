<h1>Problem Statement provided by Great Learning</h1>

<h2>Data Description:</h2>
<p>
  The actual solar radiation(watt/m^2) under a specific time was determined from laboaratory. Data is in raw form(not scaled). 
 <br>
  The data has 10 quantitative input variables, and 1 quantitaive output variable and 32686 instances (observations).
</p>

<h2>Domain:</h2>
<p>Envirnomental Aspects</p>

<h2>Context:</h2>
Radiation is the most important topic for envinoment as well as for human beings, The solar radiation prediction is a highly non linear function of age and ingredients. These ingredients include UNIX time, Data, Time, Temperature, Pressure, Humidity, WindDirection(Degrees), Speed, Time Sun Rise and Time Sun Set.

<h2>Attribute Information:</h2>

* UNIXTime : Unix time is currently defined as the number of seconds which have passed since 00:00:00 UTC on Thursday, 1 January 1970, which is referred to as the Unix epoch.
* Data : measured in yyyy-mm-dd format 
* Time : measured in hh:mm:ss 24-hour format
* Radiation : measured in watts per meter^2
* Temperature : measured in degrees Fahrenheit
* Pressure : measured in Hg
* Humidity : measured in percent
* WindDirection(Degrees) : measured in degrees
* Speed : measured in miles per hour
* TimeSunRise/TimeSunSet : Hawaii time

<h2>Learning Outcomes:</h2>

* Exploratory Data Analysis
* Building ML models for regression
* Hyper parameter tuning

<h2>Objective:</h2>
Modeling of solar radiation of high performance prediction using Machine Learning

<h2>Steps and tasks:</h2>

1. Deliverable -1 (Exploratory data quality report reflecting the following)

a. Univariate analysis

i. Univariate analysis – data types and description of the independent attributes which should include (name, meaning, range of values observed, central values (mean and median), standard deviation and quartiles, analysis of the body of distributions / tails, missing values, outliers)

b. Multivariate analysis
i. Bi-variate analysis between the predictor variables and between the predictor variables and target column. Comment on your findings in terms of their relationship and degree of relation if any. Presence of leverage points. Visualize the analysis using boxplots and pair plots, histograms or density curves. Select the most appropriate attributes

c. Pick one strategy to address the presence outliers and missing values and perform necessary imputation

2. Deliverable -2 (Feature Engineering techniques)

a. Identify opportunities (if any) to create a composite feature, drop a feature etc.

b. Decide on complexity of the model, should it be simple linear
model in terms of parameters or would a quadratic or higher
degree help

c. Explore for gaussians. If data is likely to be a mix of gaussians, explore individual clusters and present your findings in terms of the independent attributes and their suitability to predict strength

3. Deliverable -3 (create the model )

a. Obtain feature importance for the individual features and present your findings

4. Deliverable -4 (Tuning the model)

a. Algorithms that you think will be suitable for this project

b. Techniques employed to squeeze that extra performance out of the model without making it overfit or underfit

c. Model performance range at 95% confidence level

<h2>Results/ Accuracy obtained in this project:</h2>
<p>
We can conclude that RMSE is low for test data so it show that it is generalized and R squared is 0.74 so we can say that 74% is the accuracy of our hyperparameter tune Random Forest Regressor model.
<br>
You get a fairly large interval from 156 to 165, and your previous point estimate of 160 is roughly in the middle of it.
<br>
If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation. That’s because your system ends up fine-tuned to perform well on the validation data and will likely not perform as well on unknown datasets. That’s not the case in this example since the test RMSE is lower than the validation RMSE, but when it happens you must resist the temptation to tweak the hyperparameters to make the numbers look good on the test set; the improvements would be unlikely to generalize to new data.
</p>

<h2>Note:</h2>

* There is one csv file for data and other two is jupyter nootebook and python file.
* You can run this code one of this file by downloading or copy it in any plateform which support python.
* You can not see the boxplot in github because it does not support plotly library but you can see it by pasting the link of my GitHub notebook into http://nbviewer.jupyter.org/.