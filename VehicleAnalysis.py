
# coding: utf-8

# # Car Evaluation Classification
# This dataset focuses on analyzing the type of car it may be based on features such as Hollows_Ratio and Circularities. 
# I used the following order of operations to evaluate my data. In the Final Results for this particular dataset we can see that the Logistic Regression Model is the best to use and performs best without using any ensemble approaches.
# 
# 1. Define Problem: Investigate and characterize the problem in order to better understand
# the goals of the project.
# 2. Analyze Data: Use descriptive statistics and visualization to better understand the data
# you have available.
# 3. Prepare Data: Use data transforms in order to better expose the structure of the
# prediction problem to modeling algorithms.
# 4. Evaluate Algorithms: Design a test harness to evaluate a number of standard algorithms
# on the data and select the top few to investigate further.
# 5. Improve Results: Use algorithm tuning and ensemble methods to get the most out of
# well-performing algorithms on your data.
# 6. Present Results: Finalize the model, make predictions and present results.

# In[112]:

import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from matplotlib import pyplot
from sklearn.ensemble import AdaBoostClassifier
import datetime as dt
import ast
from pandas import read_csv
data = pd.read_csv('/Users/jermainemarshall/Documents/vehicle.csv')
def remove_whitespace(x):
    """
    Helper function to remove any blank space from a string
    x: a string
    """
    try:
        # Remove spaces inside of the string
        x = "".join(x.split())

    except:
        pass
    return x
#to= remove_whitespace(df)
#to.OnThyroxine
data.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)


# # Understanding your data using visualizations and looking for correlations
# Some algorithms may not perform as well if attributes are highly correlated

# In[113]:

data.describe()
#From the data we can see that most of the data is on the same scale but let's normalize the data


# In[117]:

data.describe()
data.shape
#data.hist()
#plt.show()
correlations = data.corr(method='pearson')

#print(correlations)
skew = data.skew()
#print(skew)
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()


# In[66]:

array = data.values
X = array[:,0:18]
Y = array[:,18]
validation_size = 0.20
num_folds = 10
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)


# # Spot Check Multiple Algorithms
# We Can see that logistic regression has the best performance on the validation data and LDA performas well along with it. 

# In[78]:

# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=5, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train,Y_train)
    predictions = model.predict(X_validation)
    print("Accuracy on  validation set:")
    print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


# In[79]:

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # Standardizing data to see if results may change
# We can see that logistic regression and LDA are still the best performing algorithms

# In[103]:

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression())])))
pipelines.append(('ScaledLda', Pipeline([('Scaler', StandardScaler()),('LASSO',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVR', SVC())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)# scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print("Accuracy on Training Data", msg)
    model.fit(X_train,Y_train)
    predictions = model.predict(X_validation)
    print("Accuracy of",name,"on validation set:" ,accuracy_score(Y_validation, predictions))
    print('\n')
    


# In[81]:

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # Feature Selection Using Principal Component Analysis and SelectKBest

# In[82]:

features = []
features.append(('pca', PCA(n_components=13)))
features.append(('select_best', SelectKBest(k=10)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X_validation, Y_validation, cv=kfold)
print(results.mean())


# # Now lets try Ensemble methods to see if we can improve results
# Various different parameters were tried but none seem to outperform the original logistic regression

# In[25]:

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = KNeighborsClassifier()
estimators.append(('Knn', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


# In[94]:

num_trees = 15
seed=7
kfold = KFold(n_splits=5, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[49]:

from sklearn.ensemble import ExtraTreesClassifier
num_trees = 50
max_features = 7
kfold = KFold(n_splits=10, random_state=7)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[95]:

from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 114
kfold = KFold(n_splits=5, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[ ]:



