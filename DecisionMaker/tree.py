from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

target = 'popularity'
feature = ['valence','year','acousticness','danceability','duration_ms','energy','explicit','instrumentalness','key','liveness','loudness','mode','speechiness','tempo']
data_path = "../data/data_by_decade/data_from_"

def create_tree(decade = "00s"):
    myTree = tree.DecisionTreeClassifier(criterion="gini", splitter="best")
    print("1")
    print(data_path + decade + ".csv")
    df = pd.read_csv(str(data_path + decade + ".csv"))
    print("2")
    print(df)
    myTree.fit(df[feature], df[target])
    sample = [[0.613,2000,0.14300000000000002,0.843,270507,0.8059999999999999,1,0.0,4,0.0771,-5.9460000000000015,0,0.269,94.948]]
    print(sample)
    ans = myTree.predict(sample)
    print(ans)


create_tree()
