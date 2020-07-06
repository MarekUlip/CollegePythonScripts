"""
Experiments with regression.
"""
from datetime import datetime

import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def convert_dates_to_day_of_year(dates):
    d_o_months = []
    for date in dates:
        dt = datetime.strptime(date, '%Y-%m-%d')
        d_o_months.append(dt.timetuple().tm_yday)
    print(d_o_months[1000:1005])
    return d_o_months


def normalize(data_frame):
    x = data_frame.values
    return pd.DataFrame(preprocessing.normalize(x, axis=0), columns=data_frame.columns)


def test_models(data_frame: pd.DataFrame, target_column):
    y = data_frame[target_column].values
    data_frame = data_frame.drop([target_column], axis=1)
    data_frame = data_frame.values

    X_train, X_test, Y_train, Y_test = train_test_split(data_frame, y, test_size=0.2, random_state=42)
    print('lasso:')
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(X_train, Y_train)
    measure_accuracy(lasso.predict(X_test), Y_test)
    print('ridge:')
    ridge = linear_model.Ridge(alpha=0.5)
    ridge.fit(X_train, Y_train)
    measure_accuracy(ridge.predict(X_test), Y_test)
    print('Linear:')
    linear = linear_model.LinearRegression()
    linear.fit(X_train, Y_train)
    measure_accuracy(linear.predict(X_test), Y_test)

    print('svr:')
    svm = SVR(gamma='scale', C=100000)
    svm.fit(X_train, Y_train)
    measure_accuracy(svm.predict(X_test), Y_test)
    print('dtr:')
    # dec tree regressor
    dcr = DecisionTreeRegressor()
    dcr.fit(X_train, Y_train)
    measure_accuracy(dcr.predict(X_test), Y_test)

    print("rfr:")
    rfr = RandomForestRegressor(n_estimators=20)
    rfr.fit(X_train, Y_train)
    measure_accuracy(rfr.predict(X_test), Y_test)

    print('knr:')
    # nearest neigh regressor
    knr = KNeighborsRegressor(n_neighbors=10)
    knr.fit(X_train, Y_train)
    measure_accuracy(knr.predict(X_test), Y_test)

    print('nn:')
    nn = MLPRegressor(hidden_layer_sizes=[100, 100, 100], max_iter=500)
    nn.fit(X_train, Y_train)
    measure_accuracy(nn.predict(X_test), Y_test)


def measure_accuracy(predicted, targets):
    print("MSE: {}".format(mean_squared_error(targets, predicted)))
    print("MAE: {}".format(mean_absolute_error(targets, predicted)))
    print("R2: {}".format(r2_score(targets, predicted)))


df = pd.read_csv('bike.csv')
dates = df['dteday'].values
dates = convert_dates_to_day_of_year(dates)
dates = pd.DataFrame(dates, columns=['dteday'])
df['dteday'] = dates
df = df.drop(['casual', 'registered'], axis=1)
keep_back = df['cnt']
df = df.drop(['cnt'], axis=1)
df = normalize(df)
df = df.join(keep_back)

print(df)
test_models(df, 'cnt')
