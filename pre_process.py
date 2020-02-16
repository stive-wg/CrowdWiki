import pandas as pd
import glob
import json

path = r'./'
all_files = glob.glob(path + "./dataset/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame['category'] = frame['category'].apply(lambda  x: json.loads(x))
frame['category'] = frame['category'].apply(lambda x: x['name'])
frame.head()

frame = pd.read_csv("./new_data.csv")