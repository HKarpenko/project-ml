from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

target = 'popularity'
feature = ['valence','year','acousticness','danceability','duration_ms','energy','explicit','instrumentalness','liveness','loudness','mode','speechiness','tempo']
data_path = "../data/data_by_decade/data_from_"

def create_tree(decade = "00s"):
    myTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6) # the best crit & depth

    df = pd.read_csv(str(data_path + decade + ".csv"))

    X_train, X_test, y_train, y_test = train_test_split(df[feature], df[target], test_size=0.3, random_state=1)

    myTree.fit(X_train, y_train)

    ans = myTree.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, ans))


create_tree()
