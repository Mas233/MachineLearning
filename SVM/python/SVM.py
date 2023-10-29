import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def get_data():
    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    Y = [-1] * 20 + [1] * 20
    return X,Y


def get_sv_hp(X,Y):
    # 创建SVM分类器
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    # 获取支持向量
    support_vectors = clf.support_vectors_

    # 获取超平面参数
    w = clf.coef_
    b = clf.intercept_

    # 可视化数据和超平面
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格来绘制超平面
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.decision_function(xy).reshape(xx.shape)

    # 绘制超平面和间隔
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.savefig('result.png',dpi=300)
    plt.show()


if __name__=='__main__':
    x,y=get_data()
    get_sv_hp(x,y)