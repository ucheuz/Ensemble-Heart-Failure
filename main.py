import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import recall_score

# Load csv file
df = pd.read_csv("heart_failure_data_complete.csv")
data = df.to_numpy()

# Print number of samples
print('Number of samples: ',data.shape[0])

# Create and standardise the feature matrix, create label vector
X = StandardScaler().fit_transform(data[:,[1,2]])
y = data[:,0]
print('Features dims: {}  Labels dims: {}'.format(X.shape, y.shape))


def plotData(X,y):
    plt.plot(X[y==0,0],X[y==0,1],'bo', label = 'Healthy')
    plt.plot(X[y==1,0],X[y==1,1],'rd', label = 'Mild HF')
    plt.plot(X[y==2,0],X[y==2,1],'g^', label = 'Severe HF')
    plt.legend()
    plt.title('Heart failure data')
    plt.xlabel('EF')
    plt.ylabel('GLS')

def plotDecisionBoundary(model,X,y):
    # Create an array that represents the sampled feature space
    xx = np.linspace(-3, 3, 500)
    yy = np.linspace(-3, 3.5, 500).T
    xx, yy = np.meshgrid(xx, yy)
    Feature_space = np.c_[xx.ravel(), yy.ravel()]

    # predict labels
    y_pred = model.predict(Feature_space).reshape(xx.shape)

    # plot predictions
    plt.contourf(xx,yy,y_pred, cmap = 'summer')
    plotData(X,y)

def evaluate(model,X,y):
    print('Accuracy: ', np.round(model.score(X,y),2))
    print('CV accuracy: ', np.round(cross_val_score(model,X,y).mean(),2))
    y_pred_cv = cross_val_predict(model,X,y)
    print('CV recalls: ',np.round(recall_score(y, y_pred_cv, average=None),2))
    plotDecisionBoundary(model, X,y)
-------
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X,y)
evaluate(clf,X,y)
-------
from sklearn.ensemble import BaggingClassifier

# Decision tree classifier
base_clf = DecisionTreeClassifier()

# Initialize the BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=base_clf,
    n_estimators=20,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

# Fit the BaggingClassifier to the data
bagging_clf.fit(X, y)

# Evaluate the BaggingClassifier
evaluate(bagging_clf, X, y)
------
from sklearn.linear_model import LogisticRegression
clf2=LogisticRegression(C=10)
clf2.fit(X,y)
evaluate(clf2,X,y)
------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Initialize the base estimator (Logistic Regression)
base_estimator = LogisticRegression(C=10)

# Initialize the AdaBoost classifier with Logistic Regression as base estimator
ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)

# Fit the AdaBoost classifier
ada_boost.fit(X, y)

# Evaluate the AdaBoost classifier using the evaluate function
evaluate(ada_boost, X, y)
