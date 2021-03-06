# -*- coding: utf-8 -*-
"""Facebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a4gOtL3VbUv63_nUS1va01UECAReilj9
"""

import pandas as pd
df = pd.read_csv('/content/CW2_Facebook_metrics.csv')
df.head(10)

df.isna().sum()

import pandas as pd
df = pd.read_csv('/content/CW2_Facebook_metrics (1).csv')
df.head(10)

df.isna().sum()

df.isna().sum()/len(df)*100

df.dropna(axis=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/CW2_Facebook_metrics (1).csv')
df.sample(5)

df.head()

import pandas as pd
import matplotlib.pyplot as plt
def boxplot (df, ft):
    df.boxplot(coloumn=[ft])
    plt.grid(False)
    plt.show()

for feature in ['Page total likes','Category','Post Month']:
  boxplot(diabetes,feature)

# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Reading the data
df = pd.read_csv("/content/CW2_Facebook_metrics.csv")
print(df.shape)
print(df.info())

df.describe()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))

plt.boxplot(df["Page total likes","Shares"])
plt.show()

import

import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['Total Interactions'])
plt.subplot(1,2,2)
sns.distplot(df['Page total likes'])
plt.show()

print("Highest allowed",df['Total Interactions'].mean() + 3*df['Total Interactions'].std())
print("Lowest allowed",df['Total Interactions'].mean() - 3*df['Total Interactions'].std())

df[(df['Total Interactions'] > 1352.8193540950506) | (df['Total Interactions'] < -928.5793540950507)]

upper_limit = df['Total Interactions'].mean() + 3*df['Total Interactions'].std()
lower_limit = df['Total Interactions'].mean() - 3*df['Total Interactions'].std()

df['Total Interactions'] = np.where(
    df['Total Interactions']>upper_limit,
    upper_limit,
    np.where(
        df['Total Interactions']<lower_limit,
        lower_limit,
        df['Total Interactions']
    )
)

df['Total Interactions'].describe()

import pandas as pd
df = pd.read_csv('/content/CW2_Facebook_metrics (1).csv')
df.dropna(axis=1,inplace=True)
df.head(10)

df['Total Interactions'] = (df['Total Interactions']-
df['Total Interactions'].mean())/df['Total Interactions'].std()

df['Total Interactions'].min()

df['Total Interactions'].max()

import pandas as pd
# Creating pandas DataFrame
url = '/content/CW2_Facebook_metrics.csv'
ufo = pd.read_csv(url)
ufo.head()
ufo.shape
# inplace=True to effect change
ufo.drop("Type", axis=1, inplace=True)

import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

X, y = load_diabetes(return_X_y=True)

# Train classifiers
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)

ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
ereg.fit(X, y)

# Commented out IPython magic to ensure Python compatibility.
# Importing required libraries.
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
# %matplotlib inline 
sns.set(color_codes=True)

df = pd.read_csv('/content/CW2_Facebook_metrics (1).csv')
# To display the top 5 rows
df.head(5)

# To display the bottom 5 rows
df.tail(5)

# Checking the data type
df.dtypes

# Dropping irrelevant columns
df = df.drop(["Type"], axis=1)
df.head(5)

# Total number of rows and columns
df.shape
(11914, 10)
# Rows containing duplicate data
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows:", duplicate_rows_df.shape)

# Used to count the number of rows before removing the data
df.count()

# Dropping the duplicates 
df = df.drop_duplicates()
df.head(5)

# Counting the number of rows after removing duplicates.
df.count()

# Finding the null values.
print(df.isnull().sum())

# Dropping the missing values.
df = df.dropna() 
df.count()

# After dropping the values
print(df.isnull().sum())

sns.boxplot(x=df["Total Interactions"])

sns.boxplot(x=df["Post Month"])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape

# Finding the relations between the variables.
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c

# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['Total Interactions'], df['Post Month'])
ax.set_xlabel('Total Interactions')
ax.set_ylabel('Post Month')
plt.show()

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)

from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
    random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() > 0.999

import pandas as pd
#read in the data using pandas
df = pd.read_csv('/content/CW2_Facebook_metrics.csv')
#check data has been read in properly
df.head()

#check number of rows and columns in dataset
df.shape

#create a dataframe with all training data except the target column
X = df.drop(columns=['Total Interactions'])

#separate target values
y = df['Total Interactions'].values

from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=1)
print(y_test)

s = '10.5674'

f = float(s)

print(type(f))
print('Float Value =', f)

f1 = 10.23
f2 = 20.34
f3 = 30.45

# using f-string from Python 3.6+, change to format() for older versions
print(f'Concatenation of {f1} and {f2} is {str(f1) + str(f2)}')
print(f'CSV from {f1}, {f2} and {f3}:\n{str(f1)},{str(f2)},{str(f3)}')
print(f'CSV from {f1}, {f2} and {f3}:\n{", ".join([str(f1),str(f2),str(f3)])}')

import pandas as pd
import numpy as np

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(999, inplace=True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

dataset = pd.read_csv('/content/CW2_Facebook_metrics.csv')
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print (ac)

# Calculating error for K values between 1 and 40
error = []
import numpy as np
import matplotlib.pyplot as plt
# Calculating error for K values between 1 and 40
for i in range(1, 40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train, y_train)
 pred_i = knn.predict(X_test)
 error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed',
marker='o',
 markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))

from sklearn.datasets import load_digits
digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print('Image Data Shape' , digits.data.shape)
# Print to show there are 1797 labels (integers from 0???9)
print("Label Data Shape", digits.target.shape)

import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(x_test[0].reshape(1,-1))

logisticRegr.predict(x_test[0:10])

predictions = logisticRegr.predict(x_test)

# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)



from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
    random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() > 0.999

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
    loss='squared_error'
).fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))

# Step 1: Import packages, functions, and classes
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Get data
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])

# Step 3: Create a model and train it
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

# Step 4: Evaluate the model
p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print('x:', x, sep='\n')

print('y:', y, sep='\n', end='\n\n')
print('intercept:', model.intercept_)
print('coef:', model.coef_, end='\n\n')

print('p_pred:', p_pred, sep='\n', end='\n\n')

print('y_pred:', y_pred, end='\n\n')

print('score_:', score_, end='\n\n')

print('conf_m:', conf_m, sep='\n', end='\n\n')

print('report:', report, sep='\n')

plt.show()

model.predict(x)

import seaborn as sns

sns.regplot(x=x, y=y, data=df, logistic=True, ci=None)
#import dataset from CSV file on Github
url = "/content/CW2_Facebook_metrics.csv"
data = pd.read_csv(url)

#view first six rows of dataset
data[0:6]
#define the predictor variable and the response variable
x = data['Post Month']
y = data['Total Interactions']

#plot logistic regression curve
sns.regplot(x=x, y=y, data=data, logistic=True, ci=None)

cm = confusion_matrix(y, model.predict(x))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

print(clf.predict([[0, 0, 0, 0]]))



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=2, random_state=42
)

train_samples = 100  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,
    test_size=100_000 - train_samples,
)

import numpy as np

from sklearn.svm import LinearSVC


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba

model = RandomForestClassifier(n_estimators=110) 
model.fit(X_train, y_train)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
bagging_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=250,
    max_samples=100, bootstrap=True, random_state=101)
bagging_clf.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier
random_forest_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42,n_jobs=-1)
random_forest_clf.fit(X_train, y_train)
y_pred_random_forest = random_forest_clf.predict(X_test)
bagging_clf = BaggingClassifier(
    DecisionTreeClassifier(max_leaf_nodes=16),
    n_estimators=500, random_state=101)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
np.sum(y_pred_bagging == y_pred_random_forest) / len(y_pred_bagging)

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

bankdata = pd.read_csv("/content/CW2_Facebook_metrics (1).csv")

bankdata.shape

bankdata.head()

X = bankdata.drop('Total Interactions', axis=1)
y = bankdata['Total Interactions']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GaussianNB()
clf4 = SVC(probability=True, random_state=42)

eclf = VotingClassifier(estimators=[('LR', clf1), ('RF', clf2), ('GNB', clf3), ('SVC', clf4)],
                        voting='soft', weights=[1,2,1,1])

eclf.fit(X_train, y_train)

# random search logistic regression model on the sonar dataset
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
# load dataset
url = '/content/CW2_Facebook_metrics.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# grid search logistic regression model on the sonar dataset
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# load dataset
url = '/content/CW2_Facebook_metrics.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# random search linear regression model on the auto insurance dataset
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
# load dataset
url = '/content/CW2_Facebook_metrics.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# grid search linear regression model on the auto insurance dataset
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
# load dataset
url = '/content/CW2_Facebook_metrics.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
# define search
search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

from sklearn.datasets import load_breast_cancer
br_cancer = load_breast_cancer()
br_cancer.target[0:3]

X, y = br_cancer['data'], br_cancer['target']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_pred_sgd = sgd_model.predict(X_test)
y_pred_log = log_model.predict(X_test)

from sklearn.metrics import accuracy_score
knn_score = accuracy_score(y_test, y_pred_knn)
sgd_score = accuracy_score(y_test, y_pred_sgd)
log_score = accuracy_score(y_test, y_pred_log)
print("Accuracy score (KNN): ", knn_score)
print("Accuracy score (SGD): ", sgd_score)
print("Accuracy score (Logistic): ", log_score)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_knn)
cm2 = confusion_matrix(y_test, y_pred_sgd)
cm3 = confusion_matrix(y_test, y_pred_log)
print(cm1)
print(cm2)
print(cm3)

from sklearn.metrics import classification_report
targets = br_cancer.target_names
print(classification_report(y_test, y_pred_knn,
target_names=targets)) # knn classification report
print(classification_report(y_test, y_pred_sgd,
target_names=targets)) # sgd classification report
print(classification_report(y_test, y_pred_log,
target_names=targets)) # logistic classification report

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# https://scikitlearn.org/stable/modules/generated/sklearn.metrics.classification_report.html
y_true = np.array([0, 1, 2, 2, 2])
y_pred = np.array([0, 0, 2, 2, 1])
target_names = ['class 0', 'class 1', 'class 2']

report = classification_report(y_pred=y_pred, y_true=y_true,
                               target_names=target_names, output_dict=True)

#Transposed to make the metrics name header
report_df = pd.DataFrame(report).T
report_df

# index=Note that if set to False, the label name will disappear.
report_df.to_csv("/content/CW2_Facebook_metrics.csv")

pd.read_csv("/content/CW2_Facebook_metrics.csv", index_col=0)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
#Cross Validation
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
#Confusion Matrix
from sklearn.metrics import confusion_matrix
#Precision and Recall
from sklearn.metrics import precision_score, recall_score, f1_score
#ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

train = pd.read_csv('/content/CW2_Facebook_metrics.csv')
test = pd.read_csv('/content/CW2_Facebook_metrics.csv')

train.shape

some_image = X.values[36001]
>>> some_image_reshape = some_image.reshape(28,28)
>>> plt.imshow(some_image_reshape)

import pandas as pd
import numpy as np
#importing the dataset into kaggle
df = pd.read_csv("/content/CW2_Facebook_metrics.csv")

df.head()

df.drop("Type",axis=1,inplace=True)
df.drop("Lifetime Post Impressions by people who have liked your Page",axis=1,inplace=True)
df.drop("Lifetime Post reach by people who like your Page",axis=1,inplace=True)
df.drop("Lifetime People who have liked your Page and engaged with your post",axis=1,inplace=True)
df.drop("Page total likes",axis=1,inplace=True)
df.drop("Post Weekday",axis=1,inplace=True)
df.drop("Post Hour",axis=1,inplace=True)

df.head()

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)

knn_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
rmse

test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
rmse

import seaborn as sns
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap
)
f.colorbar(points)
plt.show()

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cmap
)
f.colorbar(points)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions
# Initialize SVM classifier
clf = svm.SVC(kernel='linear')
# Fit data
clf = clf.fit(X_train, y_train)
# Generate confusion matrix
matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show(matrix)
plt.show()

# Import needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df.head()

df=df.head()

# Split dataset into features and target
y = df['Total Interactions']
X = df.drop('Total Interactions', axis=1)

# View count of each class
y.value_counts()

# Instantiate and fit the RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

# Make predictions for the test set
y_pred_test = forest.predict(X_test)

# View accuracy score
accuracy_score(y_test, y_pred_test)

# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)

# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Comments', 'Likes', 'Shares']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

# View the classification report for test data and predictions
print(classification_report(y_test, y_pred_test))

# import the voting classifier
from sklearn.ensemble import VotingClassifier

# import the voting regressor
from sklearn.ensemble import VotingRegressor

# import pandas   
import pandas as pd

# load dataset
df_heart = df.head()

df_heart.info()

# remove missing values
df_heart.dropna(inplace=True)

categories = ['Comments', 'Likes', 'Shares']
df_heart = pd.concat([df_heart, pd.get_dummies(df_heart[categories])], axis=1)
df_heart = df_heart.drop(categories, axis=1)

df.head()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_pred_sgd = sgd_model.predict(X_test)
y_pred_log = log_model.predict(X_test)

from sklearn.metrics import accuracy_score
knn_score = accuracy_score(y_test, y_pred_knn)
sgd_score = accuracy_score(y_test, y_pred_sgd)
log_score = accuracy_score(y_test, y_pred_log)
print("Accuracy score (KNN): ", knn_score)
print("Accuracy score (SGD): ", sgd_score)
print("Accuracy score (Logistic): ", log_score)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_knn)
cm2 = confusion_matrix(y_test, y_pred_sgd)
cm3 = confusion_matrix(y_test, y_pred_log)
print(cm1)
print(cm2)
print(cm3)

import pandas as pd

df.head()

#check dataset size
print(df.shape)

#split data into inputs and targets
X = df.drop(columns = ['Category'])
y = df['Category']

from sklearn.model_selection import train_test_split
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, stratify=y)

print('Original DataFrame:')
print(df)
print('\n')

# Default configuration drops rows having at least 1 missing value
print('DataFrame after dropping the rows having missing values:')
print(df.dropna())

print('Original DataFrame:')
print(df)
print('\n')

# Drop all columns that have at least one missing value
print('DataFrame after dropping the columns having missing values:')
print(df.dropna(axis=1))

print('Original DataFrame:')
print(df)
print('\n')

# Drop only those rows where all the row values are null values
print('DataFrame after dropping the rows where all the values were null values:')
print(df.dropna(axis=0, how='all'))

print('Original DataFrame:')
print(df)
print('\n')

# Drop only those columns where all values are null values
print('DataFrame after dropping the columns where all the values were null values:')
print(df.dropna(axis=1, how='all'))

# Original DataFrame
print('Original DataFrame:')
print(df)
print('\n')

# Keep rows that contain at least 2 non-missing values
print('DataFrame after using the thresh function:')
print(df.dropna(axis=0, thresh=2))

print('Original DataFrame:')
print(df)
print('\n')

# Drop only those rows where the specified column has a missing value
print('DataFrame after using the subset function:')
print(df.dropna(subset=['Paid']))

print('Original DataFrame:')
print(df)
print('\n')

# Drop only those rows where the specified column has a missing value
print('DataFrame after using the subset function:')
print(df.dropna(subset=['Paid', 'Shares', 'Likes']))

"""KNN"""

#save best model
knn_best = knn_gs.best_estimator_
#check best n_neigbors value
print(knn_gs.best_params_)

"""Random Forest"""