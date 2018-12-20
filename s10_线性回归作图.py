import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/diabetes.csv'

FEAT_COLS = ['SEX', 'BMI', 'BP', 'S1', 'S2', 'S3','S4','S5','S6']


def get_pred_line(reg_model,X_train,y_train,feat):
    w = reg_model.coef_
    b = reg_model.intercept_
    plt.figure()
    plt.scatter(X_train,y_train)
    plt.plot(X_train,w*X_train+b,c = 'red')
    plt.title(feat)
    plt.show()

def main():
    disease_data = pd.read_csv(DATA_FILE)
    for feat in FEAT_COLS:
        X = disease_data[feat].values.reshape(-1,1)
        y = disease_data['Y'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=10)
        reg_model = LinearRegression()
        reg_model.fit(X_train,y_train)
        R2 = reg_model.score(X_test,y_test)
        print('拟合值是：',R2)
        get_pred_line(reg_model,X_train,y_train,feat)

if __name__ == '__main__':
    main()