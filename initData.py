import pandas as pd
import os.path as path

def init_data():
    df = pd.read_csv("data.csv")
    my_list = ["artists", "id", "name"]
    for el in my_list:
        df.pop(el)
    df.insert(0, 'ID', range(0,len(df)))
    df.to_csv('new_data.csv', index = False)
    if not path.exists('new_data.csv'):
        exit(-1)