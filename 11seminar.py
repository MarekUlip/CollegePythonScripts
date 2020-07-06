"""
Experiments with ensamble methods
"""
import csv

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

csv_file = open('results_ensemble.csv', 'a', newline='')
csv_writer = csv.writer(csv_file, delimiter=',')

df = pd.read_csv('real_data_classification_X.csv', index_col=0)
tar = pd.read_csv('real_data_classification_y.csv', index_col=0)


def test_accuracy(data_frame, targets, test_name=""):
    y = targets
    results = []
    X_train, X_test, Y_train, Y_test = train_test_split(data_frame, y, test_size=0.2, random_state=42)
    nb = GaussianNB()
    y_pred = nb.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Gaussian naive bayess is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                            measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'NB', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])
    dec_tree = DecisionTreeClassifier()
    y_pred = dec_tree.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Decission tree is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                     measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'dec_tree', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    """nn = MLPClassifier(hidden_layer_sizes=[200, 150], activation='relu', solver='adam', max_iter=15)
    y_pred = nn.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Neural Network is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                     measure_f_score(y_pred, Y_test)))
    results.append([test_name,'NN_15_ep', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])"""

    k_nn = KNeighborsClassifier()
    y_pred = k_nn.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for KNN is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                          measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'KNN', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    bagging = BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5, n_estimators=100)
    y_pred = bagging.fit(X_train, Y_train).predict(X_test)
    print("Accuracy for bagging NB is {}. F1 score is {}".format(measure_accuracy(y_pred, Y_test),
                                                                 measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'baggingNB', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    bagging = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=100)
    y_pred = bagging.fit(X_train, Y_train).predict(X_test)
    print("Accuracy for bagging baggingDT is {}. F1 score is {}".format(measure_accuracy(y_pred, Y_test),
                                                                        measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'baggingDT', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    """bagging = BaggingClassifier(MLPClassifier(hidden_layer_sizes=[200, 150], activation='relu', solver='adam', max_iter=15),max_samples=0.5,max_features=0.5,n_estimators=20)
    y_pred = bagging.fit(X_train,Y_train).predict(X_test)
    print("Accuracy for bagging NN is {}. F1 score is {}".format(measure_accuracy(y_pred, Y_test),
                                                                measure_f_score(y_pred, Y_test)))
    results.append([test_name,'baggingNN', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])"""

    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_estimators=100)
    y_pred = bagging.fit(X_train, Y_train).predict(X_test)
    print("Accuracy for bagging KNN is {}. F1 score is {}".format(measure_accuracy(y_pred, Y_test),
                                                                  measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'baggingKNN', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    boosting = AdaBoostClassifier(n_estimators=100, random_state=42)
    y_pred = boosting.fit(X_train, Y_train).predict(X_test)
    print("Accuracy for boosting AdaBoost is {}. F1 score is {}".format(measure_accuracy(y_pred, Y_test),
                                                                        measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'boostingAda', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
    y_pred = boosting.fit(X_train, Y_train).predict(X_test)
    print("Accuracy for boosting GradientBoost is {}. F1 score is {}".format(measure_accuracy(y_pred, Y_test),
                                                                             measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'gradientBoost', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    mx_features = 15
    dec_tree = RandomForestClassifier(n_estimators=100, max_features=mx_features)
    y_pred = dec_tree.fit(X_train, Y_train).predict(X_test)
    print('Accuracy for Random forest is {}. F1 score is {}'.format(measure_accuracy(y_pred, Y_test),
                                                                    measure_f_score(y_pred, Y_test)))
    results.append([test_name, 'r_forest_15', measure_accuracy(y_pred, Y_test), measure_f_score(y_pred, Y_test)])

    csv_writer.writerows(results)
    csv_file.flush()


def measure_accuracy(predicts, targets):
    return (predicts == targets).sum() / len(predicts)


def measure_f_score(predicts, targets):
    return f1_score(targets, predicts)


test_accuracy(df, tar['0'].values)
