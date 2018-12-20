import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


DATA_FILE = '/Users/likaizhe/Desktop/Python/Machine_learning/data_ai/diabetes.csv'

FEAT_COLS = ['SEX', 'BMI', 'BP', 'S1', 'S2', 'S3','S4','S5','S6']

def main():
    disease_data = pd.read_csv(DATA_FILE)
    try_s=disease_data['SEX'].values.reshape(-1,1)
    print(try_s.shape)
    print(try_s)

    X = disease_data[FEAT_COLS].values
    y = disease_data['Y'].values
    print(y.shape)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=10)
    reg_model = LinearRegression()
    reg_model.fit(X_train,y_train)
    R2 = reg_model.score(X_test,y_test)
    print('拟合值是：',R2)


if __name__ == '__main__':
    main()

