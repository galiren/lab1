import pandas as pd
from test import id3_test

if __name__ == "__main__":
    from data_process import generate_dataframe
    data = generate_dataframe()
    data['age_group'] = pd.cut(data['age'], bins=[0, 3, 6, 15, 19, 28, 35, 45, 60, 80, 120],
                               labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], right=False)
    data.drop(['age'], axis=1, inplace=True)
    data.astype(int)

    data = data.sample(frac=1)

    X_train = data.drop(['survived'], axis=1, inplace=False)[:500]
    y_train = data['survived'][:500]
    # y_train = y_train.reshape(y_train.size, 1)
    X_test = data.drop(['survived'], axis=1, inplace=False)[500:]
    y_test = data['survived'][500:]
    # y_test = y_test.reshape(y_test.size, 1)

    # 需要将其转换为适合 DecisionTree 的数据 shape
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_train = y_train.reshape(y_train.size, 1)
    y_test = y_test.to_numpy()
    y_test = y_test.reshape(y_test.size, 1)

    id3_test(X_train, y_train, X_test, y_test)
