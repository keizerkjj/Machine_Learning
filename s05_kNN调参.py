import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import ai_utils

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def investigate_knn(iris_data,sel_cols,k_val):
    X = iris_data[sel_cols].values
    y = iris_data['label'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 10)
    knn_model = KNeighborsClassifier(n_neighbors = k_val)
    knn_model.fit(X_train,y_train)
    accuracy = knn_model.score(X_test,y_test)
    print('k = {}, 准确率是:{:.2f}'.format(k_val,accuracy*100))
    ai_utils.plot_knn_boundary(knn_model, X_test, y_test,
                               'Sepal Length vs Sepal Width, k={}'.format(k_val),
                               save_fig='sepal_k={}.png'.format(k_val))


def main():
    iris_data = pd.read_csv(DATA_FILE,index_col='Id')
    iris_data['label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)
    k_vals = [3,5,10]
    sel_cols = ['SepalLengthCm', 'SepalWidthCm']
    for k_val in k_vals:
        investigate_knn(iris_data,sel_cols,k_val)

if __name__ == '__main__':
    main()