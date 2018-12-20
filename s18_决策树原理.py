import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def main():
    iris_data = pd.read_csv(DATA_FILE,index_col='Id')
    #iris_data['label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)
    #另外一种方式来映射
    iris_data['label'] = iris_data['Species'].apply(lambda category_name:SPECIES_LABEL_DICT[category_name])
    X = iris_data[FEAT_COLS].values
    y = iris_data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)

    #对于决策树，主要的参数是max_depth
    max_depth_list = [2,3,4]
    for max_depth in max_depth_list:
        dt_model = DecisionTreeClassifier(max_depth=max_depth)
        dt_model.fit(X_train,y_train)
        train_score = dt_model.score(X_train,y_train)
        test_score = dt_model.score(X_test,y_test)
        print('实验数据的准确度是:{:.2f}%'.format(train_score*100))
        print('测试数据的准确度是:{:.2f}%'.format(test_score*100))
        print( )
if __name__ == '__main__':
    main()

