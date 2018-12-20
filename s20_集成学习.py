import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


data_file = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/wine_quality.csv'

def main():
    wine_data = pd.read_csv(data_file)
    wine_data.loc[wine_data['quality']<=5,'quality'] = 0
    wine_data.loc[wine_data['quality']>=6,'quality'] = 1

    all_cols = wine_data.columns.tolist()
    col_names = all_cols[:-1]

    X = wine_data[col_names].values
    y = wine_data['quality'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state=10)

    scaler = MinMaxScaler()
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)

    clf1 = DecisionTreeClassifier(max_depth=10)
    clf2 = LogisticRegression(C=0.1)
    clf3 = SVC(kernel='linear',probability=True)
    print(type(clf1))
    # 生成元祖。集成学习模型传入的格式应该是元祖
    clfs = [('决策树',clf1),('逻辑回归',clf2),('支持向量机',clf3)]

    for clf_tuple in clfs:
        clf_name,clf = clf_tuple
        clf.fit(X_train_scl,y_train)
        print('{}的准确率是:{:.2f}%'.format(clf_name,clf.score(X_test_scl,y_test)* 100))

    # 集成学习
    # hard voting
    hard_clf = VotingClassifier(estimators=clfs,voting='hard')
    hard_clf.fit(X_train_scl,y_train)
    acc = hard_clf.score(X_test_scl, y_test)
    print('集成学习的准确率是:{:.2f}%'.format(acc * 100))
    #soft voting
    soft_clf = VotingClassifier(estimators=clfs, voting='soft')
    soft_clf.fit(X_train_scl, y_train)
    acc = soft_clf.score(X_test_scl, y_test)
    print('集成学习的准确率是:{:.2f}%'.format(acc * 100))



if __name__ == '__main__':
    main()



