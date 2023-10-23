import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def decision_tree(max_depth=10,random_state=42):
    print(f'Generating DTree on Titanic with max_depth={max_depth} and random_state={random_state}. \n')
    # data loading
    titanic_data=pd.read_csv('data/titanic3.csv')

    # data cleaning
    titanic_data=titanic_data.dropna(subset=["age","sibsp","parch"])

    # select relevant features and target
    X=titanic_data[["pclass", "sex", "age", "sibsp", "parch"]]
    Y=titanic_data["survived"]

    # handle categorical data
    X=pd.get_dummies(X,columns=["sex"],drop_first=True)

    # split data into training & testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state)
    print('Done data preparation. \n')

    '''decision tree implementation'''
    # creating classifier
    classifier=DecisionTreeClassifier(criterion="entropy",random_state=random_state,max_depth=max_depth)
    # fitting data
    classifier.fit(X_train,Y_train)
    print('Done generating DTree. \n')
    # predicting
    y_pred=classifier.predict(X_test)

    # assessing on Micro-F1 and Macro-F1
    micro_f1=f1_score(Y_test,y_pred,average="micro")
    macro_f1=f1_score(Y_test,y_pred,average="macro")
    # visualize assessment
    metrics=['Micro-F1','Macro-F1']
    scores=[micro_f1,macro_f1]
    plt.figure(figsize=(8,6))
    plt.bar(metrics,scores,color=['#78eded','#f2e079'])
    plt.title(f"Decision Tree: max_depth={max_depth}, random_state={random_state}", color='#DC143C')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    for i, score in enumerate(scores):
        plt.text(i, score, f'{score:.5f}', ha='center', va='bottom', fontsize=12)
    plt.savefig(f"result/score_dep={max_depth}_rand={random_state}.png",dpi=300)
    print('Done visualizing assessment. \n')

    # tree visualization
    plt.figure(figsize=(40+2*max_depth,30+max_depth))
    plot_tree(classifier,filled=True,fontsize=40-2*max_depth,feature_names=X.columns,class_names=["Not Survived","Survived"])
    plt.savefig(f"result/tree_dep={max_depth}_rand={random_state}.png",dpi=300)
    print('Done visualizing Dtree. \n')
    return micro_f1,macro_f1


def depth_range_dtree_compare(lower_bound=3,upper_bound=12,random_state=42):
    index=[]
    micro_scores=[]
    macro_scores=[]
    for depth in range(lower_bound,upper_bound+1):
        micro_score,macro_score=decision_tree(max_depth=depth,random_state=random_state)
        index.append(depth)
        micro_scores.append(micro_score)
        macro_scores.append(macro_score)
    print('Done generating all trees. \n')
    plt.figure(figsize=(8,6))
    plt.plot(index,micro_scores,label='Micro-F1',marker='o',color='#00BFFF')
    plt.plot(index,macro_scores,label='Macro-F1',marker='*',color='#228B22')
    plt.title(f"Decision Tree: max_depth={lower_bound}~{upper_bound}, random_state=42")
    plt.xlabel('max_depth')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig(f"result/compare_{lower_bound}_to_{upper_bound}.png",dpi=300)
    print('Done visualizing comparison. \n')

if __name__ == '__main__':
    depth_range_dtree_compare(4,12,random_state=41)