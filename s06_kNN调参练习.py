import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_file_path = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/fruit_data.csv'
FEAT_cols = ['mass','width','height','color_score']
label_dict = {
    'apple': 0,
    'mandarin':1,
    'orange': 2,
    'lemon' :3
}

def investigate_knn(fruit_data,k_val):
    X = fruit_data[FEAT_cols].values
    y = fruit_data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=20)
    print('原始数据集共{}个样本，其中训练集样本数为{}，测试集样本数为{}'.format(
        X.shape[0], X_train.shape[0], X_test.shape[0]))
    kNNmodel = KNeighborsClassifier(n_neighbors = k_val)
    kNNmodel.fit(X_train, y_train)
    accuracy_rate = kNNmodel.score(X_test, y_test)
    print('k={},准确率是{:.2f}%'.format(k_val,accuracy_rate * 100))

def main():
    fruit_data = pd.read_csv(data_file_path)
    # 定义X和y
    fruit_data['label'] = fruit_data['fruit_name'].map(label_dict)
    k_vals = [1,3,5,7]
    for k_val in k_vals:
        investigate_knn(fruit_data,k_val)

if __name__ == '__main__':
    main()


