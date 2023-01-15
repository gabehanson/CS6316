import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
warnings.filterwarnings("ignore")


#read in dataset1
dataset1 = pd.read_csv('project3_dataset1.txt', sep='\t', header = None)

#extract feature columns
dataset1_X = dataset1.iloc[:,0:-1]
#apply min-max normalization to feature columns
dataset1_X = (dataset1_X-dataset1_X.min())/(dataset1_X.max()-dataset1_X.min())

#extract label columns
dataset1_y = dataset1.iloc[:,-1]


#read in dataset 2 as a pandas dataframe
dataset2 = pd.read_csv('project3_dataset2.txt', sep='\t', header = None)

#extract feature columns from 
dataset2_X = dataset2.iloc[:,0:-1]
#Change column 4 so present=1, absent = 0
new_col4 = []
for x in dataset2_X.iloc[:,4]:
    if x == 'Absent':
        x = 0
    elif x == 'Present':
        x = 1
    new_col4.append(x)
dataset2_X.iloc[:,4] = new_col4
#apply min-max normalization to feature columns
dataset2_X = (dataset2_X-dataset2_X.min())/(dataset2_X.max()-dataset2_X.min())

#Extract labels
dataset2_y = dataset2.iloc[:,-1]


scoring = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

def recall(TP,FN):
    recall = (TP)/(TP+FN)
    return recall

def precision(TP,FP):
    precision = (TP)/(TP+FP)
    return precision

def fmeasure(precision,recall):
    fmeasure = ((2*precision*recall)/(precision+recall))
    return fmeasure
    




#DATASET 1
print('DATASET 1')
print('')
print('')

print('BASE MODEL DEFAULT PARAMETERS')
print('--------------------------')

#separate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset1_X, dataset1_y, test_size = 0.25, random_state = 0)

#perform cross validation on training set
parameters = {}
model = RandomForestClassifier(random_state=0)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_base = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('')
print('')
testing_error_base = 1-score

accuracy_base = score



#BASE MODEL, PARAMETERS SPECIFIED BASED ON BEST PARAMETERS DETERMINED USING THE DECISION TREE CLASSIFIER
print('BASE MODEL, PARAMETERS SPECIFIED BASED ON BEST PARAMETERS DETERMINED USING THE DECISION TREE CLASSIFIER')
print('--------------------------')

#perform cross validation on training set
parameters = {}
model = RandomForestClassifier(random_state=0, criterion = 'entropy', max_depth = 25, min_samples_split = 4)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_decis = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('')
print('')

testing_error_decis = 1-score

accuracy_decis = score




## GRIDSEARCHCV TO OPTIMIZE FOR N_EST PARAMETERS 
print('GRIDSEARCHCV TO OPTIMIZE FOR N_EST PARAMETERS ')
print('--------------------------')


#perform cross validation on training set
parameters = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450]}
model = RandomForestClassifier(random_state=0)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_nest = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('--------------------------')
params = gs.best_params_
print("Best n_estimators:", str(params['n_estimators']))

print('')
print('')

testing_error_nest = 1-score

accuracy_nest = score


#GRIDSEARCHCV TO TUNE N_ESTIMATORS, CRITERION, MAX_DEPTH, BOOTSTRAP, MIN_SAMPLES_SPLIT
print('GRIDSEARCHCV TO TUNE N_ESTIMATORS, CRITERION, MAX_DEPTH, BOOTSTRAP, MIN_SAMPLES_SPLIT')
print('--------------------------')

#perform cross validation on training set
parameters = {'n_estimators': [50, 100, 150, 200, 250], 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': [2,5,10,20,50,100,150], 'bootstrap':[True, False], 'min_samples_split':[2,3,4,5,6]}
model = RandomForestClassifier(random_state=0)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_complex = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('--------------------------')
params = gs.best_params_
print("Best n_estimators:", str(params['n_estimators']))
print("Best criterion:", str(params['criterion']))
print("Best max_depth:", str(params['max_depth']))
print("Best min_samples_split:", str(params['min_samples_split']))
print("Best bootstrap:", str(params['bootstrap']))

print('')
print('')

testing_error_complex = 1-score

accuracy_complex = score

### BIAS VARIANCE TRADEOFF
labels = ['No tuning', 'Parameters From Decision Tree', 'Tuned n_est', 'Tuned Criteron, max_depth, n_estimators, \n bootstrap, min_samples_split']
training_errors =[training_error_base, training_error_decis, training_error_nest, training_error_complex]
testing_errors = [testing_error_base, testing_error_decis, testing_error_nest, testing_error_complex]
diff_base = testing_error_base-training_error_base
diff_decis = testing_error_decis-training_error_decis
diff_nest = testing_error_nest-training_error_nest
diff_complex = testing_error_complex-training_error_complex

x = np. arange(len(labels))
width = 0.2

fig, ax = plt.subplots(2, figsize=(7,8))
rects1 = ax[0].bar(x-width, training_errors, width, label="Training Error", edgecolor='k')
rects2 = ax[0].bar(x, testing_errors, width, label="Testing Error", edgecolor='k')
rects3 = ax[0].bar(x+width, [diff_base, diff_decis, diff_nest, diff_complex], width, label="Difference in Training and Testing Error", edgecolor='k')

ax[0].set_ylabel("Error")
ax[0].set_xticks(x, labels, rotation = 20, ha='right')
#ax.set_xticklabels(labels, rotation = 30, ha='right')
ax[0].legend(['Training Error', 'Testing Error', 'Difference in Training and Testing Error'])

ax[1].bar(x,[accuracy_base, accuracy_decis, accuracy_nest, accuracy_complex], edgecolor = 'k')
ax[1].set_ylabel("Accuracy")
ax[1].set_xticks(x, labels, rotation=20, ha='right')
plt.ylim([0.8,.99])
#plt.show()
















## DATASET 2
print('DATASET 2')
print('')
print('')

print('BASE MODEL DEFAULT PARAMETERS')
print('--------------------------')

#separate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset2_X, dataset2_y, test_size = 0.25, random_state = 0)

#perform cross validation on training set
parameters = {}
model = RandomForestClassifier(random_state=0)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_base = 1-score

print('--------------------------')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


print('')
print('')

testing_error_base = 1-score

accuracy_base = score

print('BASE MODEL, PARAMETERS SPECIFIED BASED ON BEST PARAMETERS DETERMINED USING THE DECISION TREE CLASSIFIER')
print('--------------------------')

#perform cross validation on training set
parameters = {}
model = RandomForestClassifier(random_state=0, criterion = 'entropy', max_depth = 5, min_samples_split = 2)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_decis = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('')
print('')
testing_error_decis = 1-score

accuracy_decis = score




## GRIDSEARCHCV TO OPTIMIZE FOR N_EST PARAMETERS 
print('GRIDSEARCHCV TO OPTIMIZE FOR N_EST PARAMETERS ')
print('--------------------------')


#perform cross validation on training set
parameters = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450]}
model = RandomForestClassifier(random_state=0)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_nest = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('--------------------------')
params = gs.best_params_
print("Best n_estimators:", str(params['n_estimators']))

print('')
print('')


testing_error_nest = 1-score

accuracy_nest = score




#GRIDSEARCHCV TO TUNE N_ESTIMATORS, CRITERION, MAX_DEPTH, BOOTSTRAP, MIN_SAMPLES_SPLIT
print('GRIDSEARCHCV TO TUNE N_ESTIMATORS, CRITERION, MAX_DEPTH, BOOTSTRAP, MIN_SAMPLES_SPLIT')
print('--------------------------')

#perform cross validation on training set
parameters = {'n_estimators': [50, 100, 150, 200, 250], 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': [2,5,10,20,50,100,150], 'bootstrap':[True, False], 'min_samples_split':[2,3,4,5,6]}
model = RandomForestClassifier(random_state=0)
gs = GridSearchCV(model, parameters, cv = 10)
gs.fit(X_train, y_train)

predictions = gs.predict(X_train)
score = gs.score(X_train, y_train)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_train, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_train, predictions)
auc = metrics.roc_auc_score(y_train, predictions)


print("Results of 10-fold Cross Validation on Training Set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Training Error: " + str(1-score))
print("AUC: "+str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''


training_error_complex = 1-score

print('')


#testing results
predictions = gs.predict(X_test)
score = gs.score(X_test, y_test)

#determining number of true neg, true pos, false pos, false neg
cm = confusion_matrix(y_test, predictions)
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

#metrics for roc curve
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
auc = metrics.roc_auc_score(y_test, predictions)


print("Results on held out test set")
print("Accuracy: " + str(score))
p = precision(TP,FP)
r = recall(TP,FN)
print("Precision: " + str(p))
print("Recall: " + str(r))
print("F1 Measure: " + str(fmeasure(p,r)))
print("Testing Error: " + str(1-score))
print("AUC: "+ str(auc))

#create roc curve
'''plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()'''

print('--------------------------')
params = gs.best_params_
print("Best n_estimators:", str(params['n_estimators']))
print("Best criterion:", str(params['criterion']))
print("Best max_depth:", str(params['max_depth']))
print("Best min_samples_split:", str(params['min_samples_split']))
print("Best bootstrap:", str(params['bootstrap']))


testing_error_complex = 1-score

accuracy_complex = score


## BIAS VARIANCE TRADEOFF

labels = ['No tuning', 'Parameters From Decision Tree', 'Tuned n_est', 'Tuned Criteron, max_depth, n_estimators, \n bootstrap, min_samples_split']
training_errors =[training_error_base, training_error_decis, training_error_nest, training_error_complex]
testing_errors = [testing_error_base, testing_error_decis, testing_error_nest, testing_error_complex]
diff_base = testing_error_base-training_error_base
diff_decis = testing_error_decis-training_error_decis
diff_nest = testing_error_nest-training_error_nest
diff_complex = testing_error_complex-training_error_complex

x = np. arange(len(labels))
width = 0.2

fig, ax = plt.subplots(2, figsize=(7,8))
rects1 = ax[0].bar(x-width, training_errors, width, label="Training Error", edgecolor='k')
rects2 = ax[0].bar(x, testing_errors, width, label="Testing Error", edgecolor='k')
rects3 = ax[0].bar(x+width, [diff_base, diff_decis, diff_nest, diff_complex], width, label="Difference in Training and Testing Error", edgecolor='k')

ax[0].set_ylabel("Error")
ax[0].set_xticks(x, labels, rotation = 20, ha='right')
#ax.set_xticklabels(labels, rotation = 30, ha='right')
ax[0].legend(['Training Error', 'Testing Error', 'Difference in Training and Testing Error'])

ax[1].bar(x,[accuracy_base, accuracy_decis, accuracy_nest, accuracy_complex], edgecolor = 'k')
ax[1].set_ylabel("Accuracy")
ax[1].set_xticks(x, labels, rotation=20, ha='right')
plt.ylim([0,0.8])
#plt.show()



















