import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def main():
    iris_data = pd.read_csv(DATA_FILE,index_col='Id')
    iris_data['label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)
    X = iris_data[FEAT_COLS].values
    y = iris_data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=10)
    classifier_dict = {
        'kNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(C=1e3),
        'SVM':SVC(C=1e3)
    }
    for model_name,model in classifier_dict.items():
        model.fit(X_train,y_train)
        acc = model.score(X_test,y_test)
        print('{}模型的准确率是:{:.2f}%'.format(model_name,acc * 100))


if __name__ == '__main__':
    main()