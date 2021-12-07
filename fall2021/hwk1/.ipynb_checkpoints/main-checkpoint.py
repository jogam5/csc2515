'''
Homework 1 Code

Team: 
Yash Prakash 1006657976
Gabriel Islas  1008098167
Charita Koya 1000433140

'''

import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# criteria for splits
SPL_CRITERIA = ['gini', 'entropy']

# data paths
CLEAN_REAL_PATH = "data/clean_real.txt"
CLEAN_FAKE_PATH = "data/clean_fake.txt"

# class labels
REAL_LABEL = '1'
FAKE_LABEL = '0'

# test data paths
TEST_DATA_X = 'data/data_test_X.csv'
TEST_DATA_y = 'data/data_test_y.csv'


# best params found after training and validating decision tree
BEST_PARAMS = {
    'max_depth' : 16,
    'criteria' : 'entropy',
}

"""
Function: load_data()
Usage: X_train, X_val, X_test, y_train, y_val, y_test, 
                            count_total, vectorizer = load_data()
-----
Returns the training, validation and test set in a 70%, 15% and 15%
allocation respectively. It preprocesses the data using a 
CountVectorizer as a feature extraction mechanism.
"""

def load_data():
    '''
    ---> Question 3a.
    ''' 

    print("Loading data...")
    f = open(CLEAN_REAL_PATH, "r")
    real_headlines = f.read().splitlines()
    f.close()

    f = open(CLEAN_FAKE_PATH, "r")
    fake_headlines = f.read().splitlines()
    f.close()

    print("All data loaded.")

    count_real = len(real_headlines)
    count_fake = len(fake_headlines)
    count_total = count_real + count_fake
    all_headlines = np.asarray(real_headlines + fake_headlines)

    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(all_headlines)

    real_labels = np.full((count_real, 1), REAL_LABEL)
    fake_labels = np.full((count_fake, 1), FAKE_LABEL)
    all_labels = np.append(real_labels, fake_labels)

    
    b = all_labels.reshape(1, count_total)
    y = b.T

    # 70 / 30 split into train + (test+val)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
        test_size=0.3, random_state=1)

    # then split 30 into 15 validation, 15 test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
        test_size=0.5, random_state=1)


    return X_train, X_val, X_test, y_train, y_val, y_test, count_total, vectorizer




def fit_tree(params, X_train, y_train, X_val, y_val):
    """helper function for 3b, to fit the decision tree with a specific criteria and depth"""
    
    clf = DecisionTreeClassifier(
            max_depth=params["max_depth"], 
            criterion=params["criteria"],
            splitter="random",
    )
    # train 
    clf.fit(X=X_train, y=y_train) 

    # validate 
    y_pred = clf.predict(X=X_val)
    correct = sum(i == j for i, j in zip(y_pred, y_val))
    score = correct / y_val.shape[0]
    print(f"max_depth: {params['max_depth']} | criteria: {params['criteria']} | score: {score}")

    return score, clf



"""
Function: select_tree_model()
Usage: best_tree = select_tree_model()
-----
Returns the best tree model considering different depths (specified in
BEST PARAMS) and also the two different criteria that SciKit provides 
to evaluate the split in a node: Gini and Entropy (defined in 
SPL_CRITERIA). Both PARAMS and SPL_CRITERIA are manually introduced 
at the top of this file.
"""
def select_tree_model(X_train, y_train, X_val, y_val, max_depth):
    '''
    ---> Question 3b
    '''

    best_score = -1
    best_tree = None
    
    # make different params 
    params = []
    for i in range(1, max_depth+1):
        for criteria in SPL_CRITERIA:
            params.append({
                "max_depth": i,
                "criteria": criteria 
            })
    
    # fit and validate on each param
    for param in params:
        score, clf = fit_tree(param, X_train, y_train, X_val, y_val)
        if (score > best_score):
            best_score = score
            best_tree = clf

    print(f"Best hyperparameters are: max_depth = {best_tree.max_depth}, criteria = {best_tree.criterion}, score = {best_score}")

    return best_tree



def test_model(clf, X_test, y_test):
    '''
    ---> Question 3c: accuracy of the decision tree
    '''

    y_out = clf.predict(X_test)
    
    # accuracy 
    return accuracy_score(y_test, y_out)




def visualize(clf, vectorizer):
    '''
    ---> Plotting the first 2 layers in Question 3c
    '''

    # plt.figure(figsize=(10,10)) 
    tree.plot_tree(clf, max_depth=2, filled = True, fontsize=8, feature_names=vectorizer.get_feature_names_out())
    plt.savefig('ree_depth_2_v2.png', bbox_inches='tight')
    plt.show()



# TODO: Paste question 3d code here
def findIndexOfFeature(feature, vectorizer):
    """helper function for question 3d to return feature names """

    featureNames = vectorizer.get_feature_names()
    return featureNames.index(feature)

"""
Function: compute_information_gain(clf, feature, vectorizer)
Usage: IG = compute_information_gain(DecisionTreeClassifier, feature, Vectorizer)
-----
Returns the information gain of splitting the tree on the 
given feature. It first computes the entropy before the split and it then
adds the entropy of the left and right nodes after the split.
"""
def compute_information_gain(clf, featureToken, vectorizer):
    """
    ---> Question 3d
    """

    # Search for feature_id
    feature_id = findIndexOfFeature(featureToken, vectorizer)
    
    tree_features = clf.tree_.feature
    num_features = clf.tree_.n_features
    node_samples = clf.tree_.weighted_n_node_samples
    feature_importance = np.zeros((num_features,)) # Feature importance as an array
    left_branch = clf.tree_.children_left # Index of features for left branch
    right_branch = clf.tree_.children_right # Index of features for right branch

    # Impurity is interpreted as Entropy
    entropy = clf.tree_.impurity    

    for nodeIndex,node in enumerate(clf.tree_.feature):
        if node >= 0:
            # Accumulate the feature importance over the nodes where it's used
            entropy_before_split = entropy[nodeIndex]*node_samples[nodeIndex]
            entropy_left_branch = entropy[left_branch[nodeIndex]]*node_samples[left_branch[nodeIndex]]
            entropy_right_branch = entropy[right_branch[nodeIndex]]*node_samples[right_branch[nodeIndex]]
            
            feature_importance[node] += entropy_before_split - entropy_left_branch - entropy_right_branch

    # Total number of samples at the root node
    feature_importance = feature_importance / node_samples[0]

    information_gain = -1
    
    # Adding feature_id column
    features = tree_features[tree_features>=0] # Negative feature is a leaf node
    features_list_by_id = list(zip(features,feature_importance[features]))
    for feat in features_list_by_id:
        if feat[0] == feature_id:
            information_gain = feat[1]
    
    print('The information gain (IG) of splitting the tree by "{0}" is {1}.'.format(featureToken, information_gain))

    return information_gain


def test_knn_model(k_value, X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    this is a helper function for question 3 e to fit knn model on a particular k value
    '''

    knn_clf = KNeighborsClassifier(n_neighbors=k_value)
    knn_clf.fit(X_train, y_train.ravel())
    
    train_score = knn_clf.score(X_train, y_train)
    val_score = knn_clf.score(X_val, y_val)
    # calculate test score only for visualisation later
    test_score = knn_clf.score(X_test, y_test)
        
    return knn_clf, val_score, train_score, test_score



"""
Function: select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test)
Usage: select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test)
-----
Returns the best KNN model after evauating the dataset using a range
of values between for K. It finally plots the train, validation and test errors
vs the range of the hyperparameter K.
"""
def select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    Question ---> 3e
    '''

    best_score = -1
    test_score_to_report = -1
    best_k = -1
    best_knn_clf = -1

    train_errors = []
    val_errors = []
    test_errors = []

    for k in range(2, 20): # all k values between 1 and 20
        knn_clf, val_score, train_score, test_score = test_knn_model(k, X_train, y_train, X_val, y_val, X_test, y_test)
        val_errors.append(1-val_score)
        train_errors.append(1-train_score)
        test_errors.append(1-test_score)
        
        # check on validation accuracy only 
        if val_score > best_score:
            best_score = val_score
            test_score_to_report = test_score
            best_k = k
            best_knn_clf = knn_clf
                

    print(f"Best val score: {best_score}, and test score: {test_score_to_report} for k = {best_k}")

    # visualisation like slide 43 in lecture slides 1

    plt.plot(train_errors, label='Train')
    plt.plot(val_errors, label = "Validation")
    # plt.plot(test_errors, label="Test")
    plt.ylabel("Error")
    plt.xlabel("K Values")
    plt.legend()
    plt.savefig('knn plot.png')
    plt.show()

def main():
    ''' 
    main function to sequentially call all functions from 3a to 3e in order and produce outputs
    '''

    # 3a
    X_train, X_val, X_test, y_train, y_val, y_test, count_total, vectorizer = load_data()
    # just a little sanity check
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 

    # 3b
    clf = select_tree_model(X_train, y_train, X_val, y_val, 20)
    

    # 3c
    accuracy_score = test_model(clf, X_test, y_test)
    print(f"Test accuracy= {accuracy_score}")
    visualize(clf, vectorizer)

    # 3d 
    # trying to two keywords donald and trump
    featureIG = compute_information_gain(clf, "donald", vectorizer)
    print(featureIG)
    featureIG = compute_information_gain(clf, "trump", vectorizer)
    print(featureIG)

    # 3e
    select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test)



if __name__ == "__main__":
    main()