import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=20)
    model_dict = {
        'kNN': (KNeighborsClassifier(),{'n_neighbors':[3,5,7], 'p':[1,2]}),
        'Logistic Regression':(LogisticRegression(),{'C':[1e-2,1,1e2]}),
        'SVM':(SVC(),{'C':[1e-2,1,1e2]})
    }
    for model_name,(model,model_params) in model_dict.items():
        clf = GridSearchCV(estimator = model,param_grid=model_params,cv=3)
        clf.fit(X_train,y_train)
        best_model = clf.best_estimator_

        acc = best_model.score(X_test,y_test)
        print('{}模型的准确率是:{:.2f}%'.format(model_name,acc*100))
        print('{}模型的最优参数是{}'.format(model_name,clf.best_params_))

if __name__ == '__main__':
    main()