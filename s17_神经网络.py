import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

data_file = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/wine_quality.csv'

def main():
    wine_data = pd.read_csv(data_file)
    #sns.countplot(data=wine_data,x ='quality')
    #plt.show()
    wine_data.loc[wine_data['quality']<= 5,'quality'] = 0
    wine_data.loc[wine_data['quality']>=6,'quality'] = 1

    # 获取列名
    all_col =wine_data.columns.tolist()
    col_name = all_col[:-1]

    X = wine_data[col_name].values
    y = wine_data['quality'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state=10)
    scaler = MinMaxScaler()
    # 进行归一化处理
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)
    #进行神经网络归类
    #实例化模型
    mlp = MLPClassifier(hidden_layer_sizes=(50,100,50),activation='relu')
    mlp.fit(X_train_scl,y_train)
    accuracy = mlp.score(X_test_scl,y_test)
    print('准确率是{:.2f}%'.format(accuracy*100))




if __name__ == '__main__':
    main()