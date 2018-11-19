import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def main():
    iris_data = pd.read_csv(DATA_FILE,index_col='Id')
    # 通过map的方式将字典里面的种类转换成 数字
    iris_data['label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)
    # 获取数据集特征X 和 数据集标签y
    X = iris_data[FEAT_COLS].values
    y = iris_data['label'].values
    # 划分数据集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3 , random_state = 10)
    # 用kNN训练数据集
    # 使用默认值，后期探讨k对模型的影响
    kNNmodel = KNeighborsClassifier()
    kNNmodel.fit(X_train,y_train)

    #用score来测算准确率
    accuracy =kNNmodel.score(X_test,y_test)
    print('准确率是{:.2f}%'.format(accuracy * 100))

    idx = 2
    x_feat = [X_test[idx,:]]
    real_y = y_test[idx]
    pred_y =  kNNmodel.predict(x_feat)
    print('实际的值是{},预测的值是{}'.format(real_y,pred_y))


if __name__ == '__main__':
    main()
