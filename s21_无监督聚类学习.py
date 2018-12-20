from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


def main():
    digits = load_digits()
    digit_data = digits.data

    k_means = KMeans(n_clusters=10)
    k_means.fit_predict(digit_data)
    #cluster_codes_ser = pd.Series(cluster_codes).value_counts()
    #cluster_codes_ser.plot(kind='bar')
    #plt.show()
    fig,axes = plt.subplots(2,5,figsize=(10,4))
    centers =k_means.cluster_centers_.reshape(10,8,8)

    for ax, center in zip(axes.flat,centers):
        ax.set(xticks=[],yticks=[])
        ax.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

    plt.show()


if __name__ == '__main__':
    main()