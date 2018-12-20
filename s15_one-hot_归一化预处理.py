import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/house_data.csv'

# 使用的特征列
NUMERIC_FEAT_COLS = ['sqft_living', 'sqft_above', 'sqft_basement', 'long', 'lat']
CATEGORY_FEAT_COLS = ['waterfront']


def preproc_data(X_train,X_test):
    # 进行one-hot处理
    encoder = OneHotEncoder(sparse=False)
    X_train_cate = encoder.fit_transform(X_train[CATEGORY_FEAT_COLS])
    X_test_cate = encoder.transform(X_test[CATEGORY_FEAT_COLS])
    # 归一化处理
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(X_train[NUMERIC_FEAT_COLS])
    X_test_num = scaler.transform(X_test[NUMERIC_FEAT_COLS])

    X_train_proc = np.hstack((X_train_num,X_train_cate))
    X_test_proc = np.hstack((X_test_num,X_test_cate))
    return X_train_proc,X_test_proc


def main():
    house_data = pd.read_csv(DATA_FILE,usecols=NUMERIC_FEAT_COLS+CATEGORY_FEAT_COLS+['price'])
    #ai_utils.plot_feat_and_price(house_data)
    X = house_data[NUMERIC_FEAT_COLS+CATEGORY_FEAT_COLS]
    y = house_data['price']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3, random_state=10)
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train,y_train)
    R2 =linear_reg_model.score(X_test,y_test)
    print('模型的拟合值是：',R2)

    # 进行数据的预处理
    X_train_proc,X_test_proc = preproc_data(X_train,X_test)
    #这个地方注意要重新设置一个新的线性模型，否则会默认使用上面训练好的线性模型来做预测
    linear_reg_model2 = LinearRegression()
    linear_reg_model2.fit(X_train_proc,y_train)
    r2 =linear_reg_model2.score(X_test_proc,y_test)
    print('模型的拟合值是：',r2)
    print('模型提升了：{:.2f}%'.format((r2-R2)/R2 *100))

if __name__ == '__main__':
    main()
