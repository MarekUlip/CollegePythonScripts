import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn import preprocessing
from collections import Counter
from scipy import stats
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import sklearn.ensemble
import csv

csv_file = open('results_classif.csv','a',newline='')
csv_writer = csv.writer(csv_file,delimiter=',')


df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
y = df.label.values
df = df.drop(['attack_cat'], axis=1)

def normalize(data_frame):
    x = data_frame.values
    return pd.DataFrame(preprocessing.normalize(x), columns=data_frame.columns)


def standardize(data_frame):
    x = data_frame.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=data_frame.columns)


def numerize_cat_vals(data_frame, columns):
    for column in columns:
        one_hot = pd.get_dummies(data_frame[column])
        data_frame = data_frame.drop(column, axis=1)
        data_frame = data_frame.join(one_hot)
    return data_frame


def test_accuracy(data_frame, target_column, test_name,neural_network_sizes=[200, 100]):
    if type(target_column) is np.ndarray:
        y = target_column
    else:
        y = data_frame[target_column].values
        data_frame = data_frame.drop([target_column], axis=1)
        data_frame = data_frame.values
    results = []
    X_train, X_test, Y_train, Y_test = train_test_split(data_frame, y, test_size=0.2, random_state=42)
    nb = GaussianNB()
    y_pred = nb.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Gaussian naive bayess is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                            measure_f_score(y_pred, Y_test)))
    results.append([test_name,'NB',measure_accuracy(y_pred, Y_test),measure_f_score(y_pred, Y_test)])
    dec_tree = DecisionTreeClassifier()
    y_pred = dec_tree.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Decission tree is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                     measure_f_score(y_pred, Y_test)))
    results.append([test_name,'dec_tree',measure_accuracy(y_pred, Y_test),measure_f_score(y_pred, Y_test)])
    mx_features= 15 if len(X_train[0]) > 15 else len(X_train[0])
    dec_tree = sklearn.ensemble.RandomForestClassifier(n_estimators=100,max_features=mx_features)
    y_pred = dec_tree.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Random forest is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                     measure_f_score(y_pred, Y_test)))
    results.append([test_name,'r_forest_15',measure_accuracy(y_pred, Y_test),measure_f_score(y_pred, Y_test)])

    """supp_vec = SVC(gamma='auto')
    y_pred = supp_vec.fit(X_train[:50000],Y_train[:50000]).predict(X_test)
    print('Accuracy for Support Vector Machine is {}. F1 score is {}'.format(measure_accuracy(y_pred,Y_test),measure_f_score(y_pred,Y_test)))
    results.append([test_name,'svm_50k',measure_accuracy(y_pred, Y_test),measure_f_score(y_pred, Y_test)])"""

    nn = MLPClassifier(hidden_layer_sizes=[200, 150], activation='relu', solver='adam', max_iter=1)
    y_pred = nn.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Neural Network is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                     measure_f_score(y_pred, Y_test)))
    results.append([test_name,'NN_1_ep',measure_accuracy(y_pred, Y_test),measure_f_score(y_pred, Y_test)])

    nn = MLPClassifier(hidden_layer_sizes=[200, 150], activation='relu', solver='adam', max_iter=15)
    y_pred = nn.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Neural Network is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                     measure_f_score(y_pred, Y_test)))
    results.append([test_name,'NN_15_ep', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])
    csv_writer.writerows(results)
    csv_file.flush()



def measure_accuracy(predicts, targets):
    """total = 0
    for i in range(len(predicts)):
        if predicts[i] == targets[i]:
            total+=1"""
    return (predicts == targets).sum() / len(predicts)


def measure_f_score(predicts, targets):
    return f1_score(targets, predicts)

def get_good_features(data_fram:pd.DataFrame,target_column):
    y = data_fram[target_column].values
    data_frame = data_fram.drop([target_column], axis=1)
    data_frame = data_frame.values
    sel = SelectFromModel(sklearn.ensemble.RandomForestClassifier(n_estimators = 100))

    X_train, X_test, Y_train, Y_test = train_test_split(data_frame, y, test_size=0.2, random_state=42)
    sel.fit(X_train,Y_train)
    print(sel.get_support())
    data_frame = data_fram.drop([target_column], axis=1)
    data_frame = data_frame.columns[sel.get_support()]
    print(data_frame)


"""print(set(df['proto'].tolist()))
print(set(df['service'].tolist()))
print(set(df['state'].tolist()))"""
# one_hot = pd.get_dummies(df['state'])
# print(one_hot.shape)
"""print(Counter(df['proto'].tolist()))
print(Counter(df['service'].tolist()))
print(Counter(df['state'].tolist()))"""
indexes_to_test = range(12)#[6,10]
cat_vals_columns = ['proto', 'service', 'state']

# No categorical and prep
if 0 in indexes_to_test:
    print("Testing accuracy with no preprocess")
    df = df.drop(['proto', 'service', 'state'], axis=1)
    test_accuracy(df, 'label','no-prep')

# Categorical No prep
if 1 in indexes_to_test:
    print("Testing accuracy with no preprocess")
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    df = numerize_cat_vals(df, ['service', 'state'])
    test_accuracy(df, 'label','no-prep-with-cat')

# Normalization categorical
if 2 in indexes_to_test:
    print('Testing accuracy after normlaization with cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = normalize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    df = numerize_cat_vals(df, ['service', 'state'])
    test_accuracy(df, 'label','normalized-with-cats')

# Normalization no categorical
if 3 in indexes_to_test:
    print('Testing accuracy after normlaization no cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = normalize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','normalized-no-cats')

# Remove outliers with categorical
if 4 in indexes_to_test:
    print('Testing accuracy after removing outliers with cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df = numerize_cat_vals(df, ['service', 'state'])
    test_accuracy(df, 'label','no-outliers-w-cats')
# print(df[(np.abs(stats.zscore(df)) < 3).all(axis=1)].shape)

# Remove outliers no categorical

if 5 in indexes_to_test:
    print('Testing accuracy after removing outliers no cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df['label']
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    test_accuracy(df, 'label','no-outliers-no-cats')

# Remove outliers and normalize no categorical
if 6 in indexes_to_test:
    print('Testing accuracy after removing outliers and normalizing no cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df['label']
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df['label']
    df = df.drop('label', axis=1)
    df = normalize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    test_accuracy(df, 'label','no-outliers-norm-no-cats')

# Remove outliers and normalize categorical

if 7 in indexes_to_test:
    print('Testing accuracy after removing outliers and nomralizing')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = normalize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    df = numerize_cat_vals(df, ['service', 'state'])
    test_accuracy(df, 'label','no-out-norm-w-cats')

# Standardization no categorical
if 8 in indexes_to_test:
    print('Testing accuracy after standardization wtih no cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','stand-no-cats')

# Standardization categorical

if 9 in indexes_to_test:
    print('Testing accuracy after standardization with cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    df = numerize_cat_vals(df, ['service', 'state'])
    test_accuracy(df, 'label','stand-w-cats')

# Remove outliers and standardize no cat
if 10 in indexes_to_test:
    print('Testing accuracy after removing outliers and standardization no cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df['label']
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df['label']
    df = df.drop('label', axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    test_accuracy(df, 'label','no-outliers-stand-no-cats')

# Remove outliers and standardize categorical
if 11 in indexes_to_test:
    print('Testing accuracy after removing outliers and stand with cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    df = numerize_cat_vals(df, ['service', 'state'])
    test_accuracy(df, 'label','no-outliers-stand-w-cats')

# PCA 5

if 12 in indexes_to_test:
    print('Testing accuracy after pca 5')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    pca = PCA(n_components=5)
    components = pca.fit_transform(df.values)
    df = pd.DataFrame(data=components)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','pca5')

# PCA 20
if 13 in indexes_to_test:
    print('Testing accuracy after pca 25')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    pca = PCA(n_components=25)
    components = pca.fit_transform(df.values)
    df = pd.DataFrame(data=components)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','pca25')

# PCA 20 with cats
if 14 in indexes_to_test:
    print('Testing accuracy after pca 25 with cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back[['service', 'state']])
    df = numerize_cat_vals(df, ['service', 'state'])
    pca = PCA(n_components=25)
    components = pca.fit_transform(df.values)
    df = pd.DataFrame(data=components)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','pca25-w-cats')

# PCA 20 removed outliers
if 15 in indexes_to_test:
    print('Testing accuracy after pca 25 removed outliers no cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df['label']
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df['label']
    df = df.drop('label', axis=1)
    df = standardize(df)
    pca = PCA(n_components=25)
    components = pca.fit_transform(df.values)
    df = pd.DataFrame(data=components)
    df = df.join(keep_back)
    test_accuracy(df, 'label','pca25-no-out')

# PCA 20 removed outliers with cats
if 16 in indexes_to_test:
    print('Testing accuracy after pca 25 removed outliers with cats')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state', 'label']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df = df.join(keep_back[['service', 'state']])
    df = numerize_cat_vals(df, ['service', 'state'])
    pca = PCA(n_components=25)
    components = pca.fit_transform(df.values)
    df = pd.DataFrame(data=components)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','pca25-no-out-w-cats')

good_labels = ['dpkts', 'sbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'dinpkt','tcprtt', 'synack', 'ackdat', 'dmean', 'ct_state_ttl', 'ct_srv_dst','label']
#Feature selection base
if 17 in indexes_to_test:
    """print('Testing accuracy after feature selection')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop(['proto', 'attack_cat'], axis=1)
    keep_back_col_names = ['service', 'state']
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    get_good_features(df,'label')"""

    print('Testing accuracy after feature selection')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df[good_labels]
    test_accuracy(df, 'label','feat-sel')

#Feature selection normalize
if 18 in indexes_to_test:
    print('Testing accuracy after feature selection with norm')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df[good_labels]
    df = df.drop(keep_back_col_names, axis=1)
    df = normalize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','feat-sel-norm')
#Feature selection standardize
if 19 in indexes_to_test:
    print('Testing accuracy after feature selection with stand')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df[good_labels]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    keep_back.reset_index(drop=True, inplace=True)
    df = df.join(keep_back['label'])
    test_accuracy(df, 'label','feat-sel-stand')

#Feature selection outliers
if 20 in indexes_to_test:
    print('Testing accuracy after feature selection no outlier')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df[good_labels]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    test_accuracy(df, 'label','feat-sel-no-outliers')

#Feature selection normalize outliers
if 21 in indexes_to_test:
    print('Testing accuracy after feature selection no outlier nomralized')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df[good_labels]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = normalize(df)
    df.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    test_accuracy(df, 'label','feat-sel-norm-no-out')

#Feature selection standardize outliers
if 22 in indexes_to_test:
    print('Testing accuracy after feature selection no outlier standardized')
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    keep_back_col_names = ['label']
    keep_back = df[keep_back_col_names]
    df = df[good_labels]
    df = df.drop(keep_back_col_names, axis=1)
    indexes = (np.abs(stats.zscore(df)) < 3).all(axis=1)
    df = df[indexes]
    df = df.join(keep_back[indexes])
    df.reset_index(drop=True, inplace=True)
    keep_back = df[keep_back_col_names]
    df = df.drop(keep_back_col_names, axis=1)
    df = standardize(df)
    df.reset_index(drop=True, inplace=True)
    df = df.join(keep_back)
    test_accuracy(df, 'label','feat-sel-no-out-stand')