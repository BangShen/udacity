#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import SelectKBest,f_classif,VarianceThreshold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

### Task 1: Select what features you'll use.
### final_features is a list of strings, each of which is a feature name.
### The first feature must be "poi".
final_features = ['poi','loan_advances','director_fees','shared_receipt_with_poi','from_this_person_to_poi',\
'from_poi_to_this_person','bonus','deferred_income','total_stock_value','expenses']

#final_features_with_new_feature = final_features + ['fraction_from_poi','fraction_this_person_to_poi']

financial_feature = ['salary','bonus','total_payments','deferral_payments','exercised_stock_options',\
                     'restricted_stock','restricted_stock_deferred','total_stock_value','expenses',\
                     'other','director_fees','loan_advances','deferred_income','long_term_incentive']
email_feature = ['from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages',\
                 'shared_receipt_with_poi']
all_features = ['poi'] + financial_feature + email_feature

# You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# selecting features by visualization of each features
data_raw = featureFormat(data_dict, all_features, sort_keys = True,remove_all_zeroes=True)

### convert array to dataframe, dataframe can be used to plot scatterplot by seaborn 
df = pd.DataFrame(data_raw,columns= all_features)
sns.pairplot(df,hue='poi')

# removing features with low variance
sel = VarianceThreshold(threshold=(0.8*(1- .8)))
print 'length of fitted feature',sel.fit_transform(data_raw[:,1:]).shape
### finding: we have 20 features in our all_features list, and chose 19 features(exclude 'poi') into sel but after selectinf\
### still with 19 features, indicating no feature was handled here

#select features automatically by selectkbest

labels_all, features_all = targetFeatureSplit(data_raw)
selector = SelectKBest(f_classif)
selector.fit(features_all,labels_all)
scores = -np.log10(selector.pvalues_)
plt.bar(range(19),scores)
plt.xticks(range(19),all_features[1:],rotation = 'vertical')
plt.title('score of each features calculated by SelectKBest')


### Task 2: Remove outliers
data_dict.pop('TOTAL')  # this is the first outlier


# find outliers by scatterplot of salary and bonus
data = featureFormat(data_dict, financial_feature)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary,bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#find financial outliers by boxplot

def boxplot_features(data_dict,feature,num):
    data = featureFormat(data_dict, feature)
    plt.figure(figsize=(6,20))
    for i in range(num):
        ax_i = plt.subplot(531+i)
        plt.boxplot(data[:,i])
        plt.xlabel(feature[i])
boxplot_features(data_dict,email_feature,4)
boxplot_features(data_dict,financial_feature[0:9],9)


# define a function to print the information of outliers of the feature

def find_outlier_key(data_dict,feature):
    #this function will return the whole record of outliers
    feature_list = []
    for key in data_dict:
        if data_dict[key][feature] == 'NaN':
            data_dict[key][feature] = 0
        feature_list.append(data_dict[key][feature])
    for key in data_dict:
        if data_dict[key][feature] == max(feature_list):
            print 'the key of this outlier is %s as he has the maximum of %s, other infor is %r' % \
            (key,feature,data_dict[key])

find_outlier_key(data_dict,'total_payments')
find_outlier_key(data_dict,'restricted_stock_deferred')
find_outlier_key(data_dict,'from_messages')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


#create two new features one is fraction_from_poi, another is fraction_this_person_to_poi
my_dataset = data_dict
def fraction_of_poi_email(poi_email,all_email):
    if all_email == 'NaN' or all_email == 0:
        fraction_poi = 0
    else:
        fraction_poi = float(poi_email)/all_email
    return fraction_poi
for key in data_dict:
    my_dataset[key]['fraction_from_poi'] = fraction_of_poi_email(my_dataset[key]['from_poi_to_this_person'],\
    my_dataset[key]['to_messages'])
    my_dataset[key]['fraction_this_person_to_poi'] = fraction_of_poi_email(my_dataset[key]['from_this_person_to_poi'],\
    my_dataset[key]['from_messages'])
# test with new features
'''
final_features = ['poi','fraction_this_person_to_poi','fraction_from_poi','loan_advances','director_fees','shared_receipt_with_poi','from_this_person_to_poi',\
'from_poi_to_this_person','bonus','deferred_income','total_stock_value','expenses']
'''
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, final_features, sort_keys = True,remove_all_zeroes=True)
labels, features = targetFeatureSplit(data)
# feature scaling process
#'''
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)
#'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#try naive_bayes
'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features,labels)
clf_pred = clf.predict(features)
print 'accuracy score is', accuracy_score(clf_pred,labels)
#0.85,0.70  0.23   0.32
'''

# try svm
'''
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state = 42, C=3)
clf.fit(features,labels)
clf_pred = clf.predict(features)
print 'accuracy score is', accuracy_score(clf_pred,labels)
'''

##try kneighborsclassifier
'''
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
clf.fit(features,labels)
clf_pred = clf.predict(features)
print 'accuracy score on training data is', accuracy_score(clf_pred,labels)
#0,88,0.86
'''
## try decisionTreeClassifier
#'''
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,max_depth=4,max_features = 3)
clf.fit(features,labels)
clf_pred = clf.predict(features)
print 'accuracy score on training data is', accuracy_score(clf_pred,labels)
#0.933,0.8666
#'''

# try adaboost
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state=0,max_depth=5,max_features = 3),\
                         n_estimators = 40,random_state = 42)
clf.fit(features,labels)
clf_pred = clf.predict(features)
print 'accuracy score on training data is', accuracy_score(clf_pred,labels)
#scores = cross_val_score(clf,features,labels)
#0.90,0.86
'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
#from sklearn.cross_validation import cross_val_score
scores = cross_val_score(clf,features,labels,cv = 10)
print 'cross validation scores:',scores.mean()


# k-fold cross validation
score_list = []
lol = KFold(len(data),10,shuffle = True)
for train_index,test_index in lol:
    feature_train = [features[i] for i in train_index]
    feature_test = [features[i] for i in test_index]
    label_train = [labels[i] for i in train_index]
    label_test = [labels[i] for i in test_index]
    
    clf.fit(feature_train,label_train)
    clf_pred = clf.predict(feature_test)
    score = accuracy_score(clf_pred,label_test)
    score_list.append(score)
print 'the mean of score by 10-folds CV for this algorithm is',np.mean(score_list)




### Task 6: Dump your classifier, dataset, and final_features so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, final_features)


#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()

    
    
