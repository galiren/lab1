import numpy as np


class ID3Node:
    # feature: 当前节点选取的用于进一步分类的特征
    # label 标签
    # child 孩子结点
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.children = {}

    def predict(self, X, features: list):
        X = np.reshape(X, (1, X.size))

        # 如果当前层已经有标签，意味着已经到达叶子
        if self.label is not None:
            return self.label

        # 否则 向下寻找
        # 先拿到选取 feature 的 index
        feature_index = list(features).index(self.feature)

        # 然后移除对应的特征
        features.remove(self.feature)
        # 根据 index 删掉最佳特征
        deleted_X = np.delete(X, feature_index, 1)
        # 排除特征的值没有在决策树中的情况，此时选取下一层的第一个特征值
        if X[0, feature_index].item() not in self.children.keys():
            return self.children[list(self.children.keys())[0]].predict(deleted_X, features=features)
        # 如果特征值存在，则进入对应的子树
        return self.children[X[0, feature_index].item()].predict(deleted_X, features=features)
# 计算条件熵
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value


# 计算选取特征前后的信息增益 (信息熵-条件熵)
def information_gain(data, labels, feature_index):
    unique_values = np.unique(data[:, feature_index])
    # 信息熵
    entropy_before_split = entropy(labels)
    total_instances = len(labels)
    # 条件熵
    weighted_entropy_after_split = 0

    for value in unique_values:
        subset_indices = np.where(data[:, feature_index] == value)[0]
        subset_labels = labels[subset_indices]
        subset_entropy = entropy(subset_labels)
        weighted_entropy_after_split += (len(subset_labels) / total_instances) * subset_entropy

    info_gain = entropy_before_split - weighted_entropy_after_split
    return info_gain


def id3(data, labels, features: list):
    if len(np.unique(labels)) == 1:  # 如果所有样本标签都一致
        return ID3Node(feature=np.unique(data), label=labels[0])

    if len(features) == 0:  # if features is empty
        return ID3Node(feature=None, label=np.argmax(np.bincount(labels[:, 0])))

    # 对特征，依次计算每一种特征对应的熵，得到最佳特征
    best_feature_index = np.argmax([information_gain(data, labels, i) for i in range(len(features))])
    best_feature = features[best_feature_index]

    # 从特征表中删除最佳特征
    remaining_features = features
    remaining_features.remove(best_feature)

    # 构造一个新结点
    node = ID3Node(feature=best_feature)

    unique_values = np.unique(data[:, best_feature_index])
    # 根据不同的特征值 分配结点
    for value in unique_values:
        subset_indices = np.where(data[:, best_feature_index] == value)[0]
        # 每个子节点对应的数据需要移除对应的特征列
        subset_data = np.delete(data[subset_indices], best_feature_index, 1)
        subset_labels = labels[subset_indices]
        if len(subset_data) == 0:  # 如果该分支下没有样本
            node.children[value] = ID3Node(label=np.argmax(np.bincount(labels)))  # 返回父节点中出现最频繁的类别
        # 有样本，则继续分裂
        else:
            node.children[value] = id3(subset_data, subset_labels, remaining_features)

    return node
