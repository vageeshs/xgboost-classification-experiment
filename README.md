## Jupiter notebook code explantion

```
from sklearn import datasets
dataset = datasets.load_iris()
```
Sklearn's datasets packages several realworld test datasets like Iris plants, handwritten digits and wine recognition for classification and Boston housing price and diabetes for regression
Iris plants dataset that we will use in this test has 3 types Setosa,Versicolour and Virginica with 4 features sepal length and width, and petal length and width

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
We do a 80:20 split of training and test data

```
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)
```
DMatrix is XGBoost's internal representation of dataset 

```
### xgboost config first iteration
config = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.1,  # learning rate 
    'verbose': 3,  # verbose
    'objective': 'multi:softprob',  # multiclass classification using the softmax objective
    'num_class': 3  # the number of classes that exist in this datset
}
boost_rounds = 30  # the number of rounds for boosting

# training and testing - numpy matrices
booster = xgb.train(config, train_data, boost_rounds)
```
In our first configuration iteration booster is configured with 
1) Random forest tree of max depth 3 
2) Learning rate of 0.1
3) Objective being multiclass classification of possible 3 types (Setosa,Versicolour and Virginica)
4) We are doing 30 iterations of training epochs

```
booster = xgb.train(config, train_data, boost_rounds)
predictions = booster.predict(test_data)
```
Train with the taining data for 30 iterations and then run the prediction for the test data. Since its a 80:20 split of training and test set, we will have 30 test sets for 150 datasets

Here are the predictions for this configuration
```
[[0.00484822 0.98747045 0.00768132]
 [0.00435326 0.03372663 0.9619201 ]
 [0.00659778 0.69444466 0.29895756]
 [0.00544722 0.98638123 0.00817155]
 [0.00485071 0.98797774 0.00717157]
....

 **Precision score of test data :  1.0**
```
