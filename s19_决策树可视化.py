import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
def plot_decision_tree(dt_model):
    tree_decision_dot = 'decision_tree_temp.dot'
    cat_name =  list(SPECIES_LABEL_DICT.keys())
    export_graphviz(dt_model,out_file=tree_decision_dot,feature_names=FEAT_COLS,class_names=cat_name,
                    filled=True,impurity=False)
    with open(tree_decision_dot) as f:
        dot_graph=f.read()
    graph = pydotplus.graph_from_dot_data(dot_graph)
    graph.write_png('tree_decision.png')

def inspect_feature_importance(dt_model):
    print('特征名称：',FEAT_COLS)
    print('重要性：',dt_model.feature_importances_)
    plt.figure()
    plt.barh(range(len(FEAT_COLS)),dt_model.feature_importances_)
    plt.yticks(np.arange(len(FEAT_COLS)),FEAT_COLS)
    plt.xlabel('Feature importances')
    plt.ylabel('Feature name')
    plt.tight_layout()
    plt.show()


def main():
    iris_data = pd.read_csv(DATA_FILE,index_col='Id')
    #iris_data['label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)
    #另外一种方式来映射
    iris_data['label'] = iris_data['Species'].apply(lambda category_name:SPECIES_LABEL_DICT[category_name])
    X = iris_data[FEAT_COLS].values
    y = iris_data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)

    dt_model = DecisionTreeClassifier(max_depth=3)
    dt_model.fit(X_train,y_train)
    test_score = dt_model.score(X_test,y_test)
    print('测试数据的准确度是:{:.2f}%'.format(test_score*100))

    plot_decision_tree(dt_model)
    inspect_feature_importance(dt_model)

if __name__ == '__main__':
    main()
