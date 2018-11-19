import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np
# 下面这个是展示数据的脚本
import ai_utils

data_file_path = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/fruit_data.csv'
FEAT_cols = ['mass','width','height','color_score']


def get_pred_label(test_sample_feat,train_data):
    dis_list =[]
    for idx, row in train_data.iterrows():
        train_feat = row[FEAT_cols].values
        dis = euclidean(train_feat,test_sample_feat)
        dis_list.append(dis)
    position = np.argmin(dis_list)
    pred_label = train_data.iloc[position]['fruit_name']
    return pred_label

def main():
    fruit_data = pd.read_csv(data_file_path)
    train_data, test_data = train_test_split(fruit_data, test_size=1 / 5, random_state=20)
    account_num = 0
    for idx,row in test_data.iterrows():
        test_sample_feat = row[FEAT_cols].values
        pred_label = get_pred_label(test_sample_feat,train_data)
        actual_label = row['fruit_name']
        print('样本{}的真实标签：{}，预测标签：{}'.format(idx, actual_label, pred_label))

        if pred_label == actual_label:
            account_num =+ 1
    # 最后算准确率
    accuracy_rate = account_num / test_data.shape[0]
    print('准确率是{:.2f}%'.format(accuracy_rate * 100))

if __name__ == '__main__':
    main()