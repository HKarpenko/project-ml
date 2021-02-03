from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
from scipy.stats import pearsonr

import pandas as pd


target = 'popularity'
feature = ['valence','acousticness','danceability','duration_ms','energy','explicit','instrumentalness','liveness','loudness','mode','speechiness','tempo']
feature_2 = ['valence','acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','mode','speechiness','tempo']
data_path = "../data/data_by_decade/data_from_10s.csv"
data_path_all = "../data/data.csv"
data_path_2 = "../data/data_2.csv"
data_path_3 = "../data/data2_preprocessed.csv"

def check_answers(answer, actual, X_test):
    answer = list(answer)
    actual = list(actual)
    ok = 0
    diff = 15
    ans_list = []
    for i in range(len(answer)):
        if(actual[i] < answer[i]+diff and actual[i] > answer[i]-diff ):
            ok+=1
            ans_list.append(1)
        else:
            ans_list.append(0)
    print(len(X_test['valence']))
    print(len(ans_list))
    X_test['answers'] = ans_list
    print(X_test.head(100))
    X_test.plot.scatter(x=feature[0],
                        y=feature[1],
                        c=X_test['answers'])
    return ok/len(answer)

def create_tree(features):
    for dp in range(1,15):
        myTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=dp) # the best crit & depth

        df = pd.read_csv(str(data_path))

        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=1)

        myTree.fit(X_train, y_train)

        ans = myTree.predict(X_test)
        print("Depth:",dp," Accuracy: ", check_answers(ans, y_test, X_test))

def create_boost(features, data_path):
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

    df = pd.read_csv(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=1)

    abc.fit(X_train, y_train)

    ans = abc.predict(X_test)
    print("Accuracy: ", check_answers(ans, y_test, X_test))

def create_forest(features, data_path):
    dp=9
    est=95

    df = pd.read_csv(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=1)

    print(type(X_test))
    forestClassifier = RandomForestClassifier(n_estimators=est, max_depth=dp, random_state=0, criterion="gini") # the best num, crit & depth (decade 10s)
    forestRegressor = RandomForestRegressor(n_estimators=est, max_depth=dp, random_state=0) # the best num, crit & depth (decade 10s)

    forestClassifier.fit(X_train, y_train)
    ansClassifier = forestClassifier.predict(X_test)
    forestRegressor.fit(X_train, y_train)
    ansRegressor = forestRegressor.predict(X_test)
    
    print("Classifier Num:", est, " Depth:", dp, " Accuracy: ", check_answers(ansClassifier, y_test, X_test))
    

    feature_classifier_importances_df = pd.DataFrame(
        {"feature": features, "importance": forestClassifier.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Display
    print(feature_classifier_importances_df)

    print("Regressor Num:", est, " Depth:", dp, " Accuracy: ", check_answers(ansRegressor, y_test, X_test))
    feature_regressor_importances_df = pd.DataFrame(
        {"feature": features, "importance": forestRegressor.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Display
    print(feature_regressor_importances_df)

"""
def features():
    classifier = RandomForestClassifier(n_estimators=95, max_depth=9, random_state=0, criterion="gini")
    feature_importances_df = pd.DataFrame(
        {"feature": feature, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=Falsevs

    # Display
    print(feature_importances_df)
"""

"""
create_forest(feature_2, data_path_2)
create_forest(feature, data_path_all)
print("BOOST")
print("feature_2")
create_boost(feature_2, data_path_2)
create_boost(feature, data_path_all)
"""
create_forest(feature_2, data_path_3)
create_forest(feature_2, data_path_all)
