import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/house_data.csv'

# 使用的特征列
FEAT_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

def plot_fitting_line(linear_reg_model,X_train,y_train,feat):
    w = linear_reg_model.coef_
    b = linear_reg_model.intercept_
    plt.figure()
    # alpha为透明度,用scatter展示出样本点
    plt.scatter(X_train,y_train,alpha=0.5)
    # 用plot制作回归线
    plt.plot(X_train,w*X_train+b,c='red')
    plt.show()



def main():
    house_data = pd.read_csv(DATA_FILE,usecols=FEAT_COLS+['price'])
    #ai_utils.plot_feat_and_price(house_data)
    for feat in FEAT_COLS:
        # 注意，这个地方，需要用reshape将行数据转换成列数据
        X = house_data[feat].values.reshape(-1,1)
        # y则不用，直接使用行数据就行
        y = house_data['price'].values
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3, random_state=10)
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train,y_train)
        R2 =linear_reg_model.score(X_test,y_test)
        print('拟合值是：',R2)
        plot_fitting_line(linear_reg_model,X_train,y_train,feat)


if __name__ == '__main__':
    main()

