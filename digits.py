import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
 

#----------FETCH MNIST Data-----------
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.data.shape, y.data.shape) 	#print dimensions of data


#----------Plot and view the images--------
plt.figure(figsize=(20,4))
for i, (X,y) in enumerate(zip(X[0:3], y[0:3])):
	plt.subplot(1, 5, i+1)
	plt.imshow(np.reshape(X, (28,28)), cmap=plt.cm.gray)
	plt.title('Training Example : %s\n' %y, fontsize = 15)
plt.show()


#-------Split data into train and test set------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7.0, random_state=0, stratify = y)  


#-------Scale data---------
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#------PCA---------
pca=PCA(0.95) 	#retain 95% variance while choosing principal components
pca.fit(X_train)
print(pca.n_components_)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


#------GridSearchCV to select optimal parameters-----
#NOTE: GridSearch CV is computationally expensive and time consuming. 
#Please comment out this section(until Evaluate Performance) and use the optimal parameters as plotted in the Readme.txt 
#unless you want to play around with different parameters.

#---Logistic Regression---
param_grid = { 
   'C': [0.001,0.01,0.1,1,10,100],
   'solver': ('saga','newton-cg','lbfgs') 	#set parameters to run cross-validation on for Logistic Regression
}
logisticRegr = LogisticRegression(multi_class = 'multinomial', tol = 0.001, penalty = 'l2', max_iter = 500, n_jobs = -1) 	#set model parameters
search = GridSearchCV(logisticRegr, param_grid, iid=False, cv=5, n_jobs= -1) 	#set gridsearch cv parameters
search.fit(X_train, y_train) 	#search for optimal parameters
print("Best parameter (CV score=%0.3f):" % search.best_score_)
Parameters_Scores_lr = pd.DataFrame(search.cv_results_) 	#store results in a pd dataframe
Parameters_Scores_lr.to_csv("GridSearch_lr_results.csv") 	
print(search.best_params_) 

#---SGDClassifier---
params_sgd = {
       'loss': ('hinge', 'log', 'perceptron'),
       'alpha': [0.1,0.01,0.001,0.0001,0.00001] 	#set parameters to run cross-validation on for Stochaistic Gradient Descent
       }
SGD = SGDClassifier(penalty='l2', early_stopping=True, max_iter=500, tol=1e-5, random_state=0, n_jobs=-1, validation_fraction=0.1) 	#set model parameters
search_sgd = GridSearchCV(SGD, params_sgd, iid=False, n_jobs= -1) 	#set gridsearch cv parameters
search_sgd.fit(X_train, y_train) 	#search for optimal parameters
print("Best parameter (CV score=%0.3f):" % search_sgd.best_score_)
Parameters_Scores = pd.DataFrame(search_sgd.cv_results_) 	#store results in a pd dataframe
Parameters_Scores.to_csv("GridSearch_sgd_results.csv")
print(search_sgd.best_params_)

#---KNN---
#params_knn = {
#        'weights': ('uniform', 'distance'),
#        'n_neighbors': [i for i in [1,5,10,15]],
#        'metric': ('euclidean', 'manhattan', 'minkowski')
#        }
#Knn = KNeighborsClassifier(algorithm = 'auto', n_jobs = -1)
#search_knn = GridSearchCV(Knn, params_knn, iid=False, n_jobs=-1)
#search_knn.fit(X_train, y_train)
#print("Best parameter (CV score=%0.3f):" % search_knn.best_score_)
#Parameters_Scores_knn = pd.DataFrame(search_knn.cv_results_)
#Parameters_Scores_knn.to_csv("GridSearch_knn_results.csv")
#print(search_knn.best_params_)

params_knn1 = {
        'n_neighbors': [i for i in [4,5,6,7,8]], 	#set parameters to run cross-validation on for Stochaistic Gradient Descent
        'metric': ('euclidean', 'minkowski')
        }
Knn1 = KNeighborsClassifier(weights = 'distance', algorithm = 'auto', n_jobs = -1) 	#set model parameters
search_knn1 = GridSearchCV(Knn1, params_knn1, iid=False, n_jobs=-1) 	#set gridsearch cv parameters
search_knn1.fit(X_train, y_train) 	#search for optimal parameters
print("Best parameter (CV score=%0.3f):" % search_knn1.best_score_)
Parameters_Scores_knn1 = pd.DataFrame(search_knn1.cv_results_) 	#store results in a pd dataframe
Parameters_Scores_knn1.to_csv("GridSearch_knn_results1.csv")
print(search_knn1.best_params_)

#---SVM---
params_svm = {
       'kernel': ('poly', 'rbf'),
       'C': [10,1,0.1,0.01,0.001],
       'degree': [2,3,4],
       'coef0': [0,1]
       }
Svm = svm.SVC(gamma = 'scale', shrinking = True, tol = 0.001, random_state = 0, decision_function_shape = 'ovr')
search_svm = GridSearchCV(Svm, params_svm, iid=False, n_jobs=-1)
search_svm.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search_svm.best_score_)
Parameters_Scores_svm = pd.DataFrame(search_svm.cv_results_)
Parameters_Scores_svm.to_csv("GridSearch_svm_results.csv")
print(search_svm.best_params_)

#--------Evaluate Performance-----

logisticRegr = LogisticRegression(C = 0.01, solver = 'newton-cg', multi_class = 'multinomial', tol = 0.001, penalty = 'l2', max_iter = 500, n_jobs = -1)
SGD = SGDClassifier(loss = 'log', alpha = 0.0001, penalty='l2', early_stopping=True, max_iter=500, tol=1e-5, random_state=0, n_jobs=-1)
SVM = svm.SVC(C = 1, coef0 = 1, degree = 4, kernel = 'poly', gamma = 'scale', shrinking = True, tol = 0.001, random_state = 0, decision_function_shape = 'ovr')

time0 = time.time()
logisticRegr.fit(X_train,y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
report = classification_report(y_test, predictions)
print('Score: %f' % (score) , time.time()-time0)
print(report)

time1 = time.time()
SGD.fit(X_train,y_train)
predictions1 = SGD.predict(X_test)
score1 = SGD.score(X_test, y_test)
report1 = classification_report(y_test, predictions1)
print('Score: %f' % (score1) , time.time()-time1)
print(report1)

time2 = time.time()
SVM.fit(X_train,y_train)
predictions2 = SVM.predict(X_test)
score2 = SVM.score(X_test, y_test)
report2 = classification_report(y_test, predictions2)
print('Score: %f' % (score2) , time.time()-time2)
print(report2)