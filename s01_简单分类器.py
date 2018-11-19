import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np
# 下面这个是展示数据的脚本
import ai_utils

data_file_path = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/Iris.csv'
Species = ['Iris-setosa',
           'Iris-versicolor',
           'Iris-virginica' ]
# 特征序列，即数据集中包含的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def get_pred_label(test_sample_feat,train_data):
    dist_list = []
    for idx,row in train_data.iterrows():
        train_feat = row[FEAT_COLS].values
        # 下面这个function是用来计算距离
        dis = euclidean(test_sample_feat,train_feat)
        dist_list.append(dis)
    #接下来在dist_list里面选出最小的值，并返回对应的label
    # pos为定位的位置，iloc找到对应的位置
    pos = np.argmin(dist_list)
    pred_label = train_data.iloc[pos]['Species']
    return pred_label



def main():
    iris_data = pd.read_csv(data_file_path,index_col='Id')
    # 将读取的数据进行plot展示
    ai_utils.do_eda_plot_for_iris(iris_data)
    # 用分类器分割数据为训练用途和测试用途
    train_data, test_data = train_test_split(iris_data,test_size=1/3, random_state=10)

    account = 0
    #用距离最短法来对比测试数据与训练数据
    #因为test_data是dataframe，所以可以得到他的index和一整行的数据（用row表示）
    for idx, row in test_data.iterrows():
        # 提取test行的特征值
        test_sample_feat = row[FEAT_COLS].values
        pred_test_label = get_pred_label(test_sample_feat,train_data)
        #真实值
        actual_value = row['Species']
        print('样本{}的真实标签{}，预测标签{}'.format(idx, actual_value, pred_test_label))

        if actual_value == pred_test_label:
            account += 1

        #测试准确率
    accuracy_rate = account / test_data.shape[0]
    print('预测准确率{:.2f}%'.format(accuracy_rate * 100))

if __name__ == '__main__':
    main()


