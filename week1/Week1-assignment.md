
# Week1 Assignment - Testing OKPy and the autograder.

For this assignment, you will load the iris flowers dataset from sklearn and predict flower species based on their petal and sepal sizes.

## Part A. Load the Iris dataset.

Sklearn comes with prepared datasets that you can load. To load the iris dataset, run:
```python
from sklearn import datasets
iris = datasets.load_iris()
```


```python
from sklearn import datasets
iris = datasets.load_iris() #loadiris is a function, call by adding parentheses, iris is the dataset
```


```python
iris.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']




```python
iris.target_names
```




    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')




```python
iris.target
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
assert (iris['data'] == datasets.load_iris()['data']).all()
```

## Part B. Use an SVM to predict iris species.

To setup the data and train an SVM, run the following:
```python
from sklearn.svm import SVC

X = iris.data
y = iris.target

classifier = SVC(kernel="rbf")
classifier.fit(X, y)
predictions = classifier.predict(X)
```


```python
from sklearn.svm import SVC #SVC = class with template for object that we're going to create; like a blueprint

X = iris.data
y = iris.target

classifier = SVC() #default; parentheses calls class, gives back class instance to variable classifier
```


```python
classifier.fit(X,y) #method = function that is bound to an object, only makes sense in context to an object
predictions = classifier.predict(X)
```

    /srv/conda/envs/notebook/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


1. Import
2. Instantiate (a class); 
- estimators:
    - general rule for classifier/
    - regresser
3. Fit
4. Predict

Overfitting: 
- tailored to specific data set, doesnt generalize well (will not perform well on future data)
- already know labels/data set
- want predictions of future data sets
- VERY accurate; might give 100% accuracy, but doesnt predict next set --> so train, test, split
- memorizes data
- ex: complex polynomials
- AKA variance

Underfitting: not specific enough
- accuracy could be better
- underpowered for problem
- misses patterns
- ex: general linear regressions
- AKA bias

**Bias variance tradeoff


```python
predictions
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
from sklearn.model_selection import train_test_split #traintestsplit is a function
```


```python
iris = datasets.loadiris()

X = iris.data
y = iris.target

classifier = train_test_split

classifier.fit(X, y)

predictions = classifier.predict(X)
```

    /srv/conda/envs/notebook/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)



```python
from sklearn.svm import SVC

# YOUR CODE HERE
raise NotImplementedError()
```


```python
assert X.shape == (150, 4)
assert len(y) == 150
```
