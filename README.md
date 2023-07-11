# PyCaret-Fintech-Churn-Prediction
this repo uses pycaret to manipulate, create and test different ML models on transactional data 

Please use this command to install the library 
pip install pycaret 

the notebook starts by installing pycaret library then we load the data and do some manipulation like removing sole traders (freelancers) and casting date column to datatime and limiting the training data to april 2023 and testing data to anything after that date 
then doing the setup stage with ignoring unimportant features and defining the churned column as target variable 
then comparing models 
we got this table comparing 14 models according to accuracy, auc, recall, precision, f1, lappa, mcc and time 
Model 	Accuracy 	AUC 	Recall 	Prec. 	F1 	Kappa 	MCC 	TT (Sec)
lightgbm 	Light Gradient Boosting Machine 	0.9360 	0.9488 	0.5711 	0.6566 	0.6108 	0.5761 	0.5778 	2.8720
gbc 	Gradient Boosting Classifier 	0.9356 	0.9473 	0.5745 	0.6520 	0.6107 	0.5758 	0.5772 	23.8120
ada 	Ada Boost Classifier 	0.9317 	0.9427 	0.5157 	0.6382 	0.5704 	0.5337 	0.5374 	7.0120
lda 	Linear Discriminant Analysis 	0.9318 	0.9375 	0.5660 	0.6233 	0.5933 	0.5561 	0.5569 	1.1760
rf 	Random Forest Classifier 	0.9303 	0.9374 	0.5398 	0.6191 	0.5767 	0.5389 	0.5404 	33.6690
et 	Extra Trees Classifier 	0.9263 	0.9189 	0.5182 	0.5928 	0.5529 	0.5130 	0.5144 	20.9110
nb 	Naive Bayes 	0.9121 	0.8422 	0.0000 	0.0000 	0.0000 	0.0000 	0.0000 	0.8280
lr 	Logistic Regression 	0.9180 	0.8305 	0.4211 	0.4533 	0.4237 	0.3896 	0.3968 	3.1120
dt 	Decision Tree Classifier 	0.9162 	0.7283 	0.4956 	0.5253 	0.5100 	0.4642 	0.4645 	2.1070
knn 	K Neighbors Classifier 	0.9148 	0.6945 	0.1474 	0.5580 	0.2332 	0.2039 	0.2559 	2.7500
qda 	Quadratic Discriminant Analysis 	0.3410 	0.5712 	0.8429 	0.1473 	0.1992 	0.0655 	0.1051 	0.8560
dummy 	Dummy Classifier 	0.9121 	0.5000 	0.0000 	0.0000 	0.0000 	0.0000 	0.0000 	0.7200
svm 	SVM - Linear Kernel 	0.9161 	0.0000 	0.0755 	0.8170 	0.1278 	0.1152 	0.2016 	2.0060
ridge 	Ridge Classifier 	0.9296 	0.0000 	0.4290 	0.6516 	0.5173 	0.4811 	0.4934 	0.6490

then the model autmatically selects the best model according to the chosen metric like auc 
and tune it 
then producing the ROC curve, feature importance and confusion matrix graphs 
then after some testing we save the model weights to be used later in future predictions. 
