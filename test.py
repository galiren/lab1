import numpy as np
from id3 import id3


def id3_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    root = id3(X_train, y_train, features=[i for i in range(X_train.shape[1])])
    y_pred = []
    for x in X_test:
        pred = root.predict(x, features=[i for i in range(X_train.shape[1])])
        y_pred.append(pred.item())

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.size, 1)
    y_res = y_pred - y_test
    # 统计没有预测准确的数目
    unmatched_counts = 0
    for y in y_res:
        unmatched_counts += y * y
    print(f'id3 预测精度: {(1.0-unmatched_counts/len(y_pred)) * 100}%')
