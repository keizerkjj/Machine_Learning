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


def main():
    fruit_data = pd.read_csv(data_file_path)
    # 定义X和y
    fruit_data['label'] = fruit_data['fruit_name'].map(label_dict)
    X = fruit_data[FEAT_cols].values
    y = fruit_data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1 / 5, random_state=20)

    kNNmodel = KNeighborsClassifier()
    kNNmodel.fit(X_train,y_train)
    accuracy_rate=kNNmodel.score(X_test,y_test)
    print('准确率是{:.2f}%'.format(accuracy_rate * 100))

if __name__ == '__main__':
    main()