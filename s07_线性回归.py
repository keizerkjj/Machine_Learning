import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import ai_utils

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/house_data.csv'

# 使用的特征列
FEAT_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']


def main():
    house_data = pd.read_csv(DATA_FILE,usecols=FEAT_COLS+['price'])
    #ai_utils.plot_feat_and_price(house_data)
    X = house_data[FEAT_COLS].values
    y = house_data['price'].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3, random_state=10)
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train,y_train)
    R2 =linear_reg_model.score(X_test,y_test)
    print('模型的拟合值是：', R2)
    
    # 利用以上训练出来的模型，来预测
    i = 10
    single_FEAT = X_test[i,:]
    y_pred = linear_reg_model.predict([single_FEAT])
    y_actual = y_test[i]
    print('样本特征:',single_FEAT)
    print(f'预测值是：{y_pred},实际值是{y_actual}')

if __name__ == '__main__':
    main()