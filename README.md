## XGBoost experiments with Iris flower dataset & explanation of jupiter notebook code

Focus is on experimenting with XGBoost configuration and understanding of change in generated random forest tree models

Sklearn's dataset package comes with several realworld test datasets like Iris plants, handwritten digits and wine recognition for classification and Boston housing price and diabetes for regression
Iris plants dataset that we will use in this test has 3 types Setosa, Versicolour and Virginica with 4 features sepal length and width, and petal length and width

##### Iris dataset #####
Sepal length | Sepal width | Petal length | Petal width	| Species
------------ | ----------- | ------------ | ----------- | --------
5.1	| 3.5 | 1.4 | 0.2 | I. setosa
4.9 | 3.0 | 1.4 | 0.2 | I. setosa
4.7 | 3.2 | 1.3 | 0.2 | I. setosa
....

```
from sklearn import datasets
dataset = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
We do a 80:20 split of training and test data

```
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)
```
DMatrix is XGBoost's internal representation of dataset 

### XGBoost 1st config attempt ###
```
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
In the 1st attempt booster is configured with 
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

```
 **Precision score of test data :  1.0**
#### Generated booster trees ####
*f0, f1, f2 and f3 are the 4 features in the dataset*

*Tree's generated in the first epoch*
```
booster[0]:
0:[f2<2.45000005] yes=1,no=2,missing=1
	1:leaf=0.284745783
	2:leaf=-0.145794407
booster[1]:
0:[f2<2.45000005] yes=1,no=2,missing=1
	1:leaf=-0.142372891
	2:[f3<1.75] yes=3,no=4,missing=3
		3:[f2<4.94999981] yes=5,no=6,missing=5
			5:leaf=0.272727281
			6:leaf=-6.50232534e-09
		4:[f2<4.85000038] yes=7,no=8,missing=7
			7:leaf=-5.10896969e-09
			8:leaf=-0.139534891
booster[2]:
0:[f2<4.75] yes=1,no=2,missing=1
	1:[f3<1.45000005] yes=3,no=4,missing=3
		3:leaf=-0.145392507
		4:[f1<2.8499999] yes=7,no=8,missing=7
			7:leaf=-5.10896969e-09
			8:leaf=-0.103448287
	2:[f3<1.75] yes=5,no=6,missing=5
		5:[f0<6.5] yes=9,no=10,missing=9
			9:leaf=0.119999997
			10:leaf=-0.0240000058
		6:[f2<4.85000038] yes=11,no=12,missing=11
			11:leaf=0.0857142806
			12:leaf=0.279069752
```
1. As you can see trees have atmost 3 levels
1. 1st tree booster[0] uses only f2 feature which is petal length (See above for Iris dataset table)
1. 2nd tree booster[1] uses f2 and f3 which is petal width
1. 3rd tree booster[2] uses all 4 features 
1. Finally softmax optimizer is applied to the leaf value deduced using these trees

*By last iteration these 3 tree weights has changed as follows* 

```
booster[87]:
0:[f2<3.4000001] yes=1,no=2,missing=1
	1:leaf=0.0432979167
	2:leaf=-0.0539936312
booster[88]:
0:[f0<5.44999981] yes=1,no=2,missing=1
	1:leaf=-0.0266337991
	2:[f2<4.85000038] yes=3,no=4,missing=3
		3:[f0<5.94999981] yes=5,no=6,missing=5
			5:leaf=0.053024698
			6:leaf=0.00710496539
		4:[f3<1.54999995] yes=7,no=8,missing=7
			7:leaf=-0.0415618606
			8:leaf=0.00824911147
booster[89]:
0:[f3<1.45000005] yes=1,no=2,missing=1
	1:leaf=-0.0434705317
	2:[f1<3.04999995] yes=3,no=4,missing=3
		3:[f3<1.75] yes=5,no=6,missing=5
			5:leaf=0.0147221982
			6:leaf=0.0579996109
		4:leaf=-0.0284879301
```

### XGBoost config 2nd iteration ###
```
config = {
    'max_depth': 2,  # the maximum depth of each tree
    .....
}
boost_rounds = 30  # the number of rounds for boosting
```
In this 2nd configuration attempt we only change the max tree depth to 2
Here are the predictions for this configuration
```
[[0.94681555 0.02785276 0.02533168]
 [0.05672611 0.2471049  0.696169  ]
 [0.02875477 0.03339498 0.93785024]
 [0.03646614 0.92628187 0.03725201]
 [0.08247369 0.42990598 0.48762035]
....
```
 **Precision score of test data :  0.9666666666666667**
1. Notice that precision score has dropped from 1.0 to 0.9666666666666667

*Tree's generated in the first epoch*
```
booster[0]:
0:[f2<2.45000005] yes=1,no=2,missing=1
	1:leaf=0.142196536
	2:leaf=-0.0729230866
booster[1]:
0:[f1<2.95000005] yes=1,no=2,missing=1
	1:[f3<1.60000002] yes=3,no=4,missing=3
		3:leaf=0.114893615
		4:leaf=-0.0646153912
	2:[f2<3.04999995] yes=5,no=6,missing=5
		5:leaf=-0.0709090978
		6:leaf=-0.00827586651
booster[2]:
0:[f3<1.6500001] yes=1,no=2,missing=1
	1:[f2<4.85000038] yes=3,no=4,missing=3
		3:leaf=-0.0728434548
		4:leaf=0.0239999983
	2:leaf=0.131360948
```
1. As you can see trees have atmost 2 levels
1. 1st tree booster[0] uses only f2 feature (petal length)
1. 2nd tree booster[1] uses f1(Sepal width), f2(petal length) and f3(petal width)
1. 3rd tree booster[2] uses only f2 and f3 feature 
1. Sepal length feature is not used

*By last iteration these 3 tree weights has changed as follows*
```
booster[87]:
0:[f2<2.45000005] yes=1,no=2,missing=1
	1:leaf=0.0433618538
	2:leaf=-0.0440045707
booster[88]:
0:[f2<5.05000019] yes=1,no=2,missing=1
	1:[f2<2.45000005] yes=3,no=4,missing=3
		3:leaf=-0.0364460312
		4:leaf=0.0234700609
	2:leaf=-0.0402130224
booster[89]:
0:[f3<1.6500001] yes=1,no=2,missing=1
	1:[f2<4.85000038] yes=3,no=4,missing=3
		3:leaf=-0.0436591133
		4:leaf=0.0148964003
	2:[f1<2.95000005] yes=5,no=6,missing=5
		5:leaf=0.0519691408
		6:leaf=0.0177164394
```