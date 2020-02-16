from flask import Flask, request, jsonify
import json
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def find_cat(blurb):
    pickle_in = open("tokenizer.pickle","rb")
    data = pickle.load(pickle_in)
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data)
    text_out = tokenizer.texts_to_sequences(blurb)
    frame = pd.read_csv("./new_data.csv")
    model = tf.keras.models.load_model("./model_v1.h5")
    return model.predict([text_out])

def piechart(category):
    out = []
    for i in frame.loc[frame['category'] == category]['country'].unique():
        out.append({
            "x":i,
            "y":frame.loc[frame['category'] == category].loc[frame['country'] == i]['backers_count'].median()
        })
    return out

def barchart(category):
    out = []
    for i in frame.loc[frame['category'] == category]['country'].unique():
        out.append({
            "x":i,
            "y":frame.loc[frame['category'] == category].loc[frame['country'] == i]['goal'].median()
        })
    return out

def top_proj(category):
    proj = []
    for i, row in frame.loc[frame['category'] == category].nlargest(3,'backers_count').iterrows():
        proj.append({
            "title":row['name'],
            "cat":row['category'],
            "des":row['blurb']
        })
    return proj


@app.route('/', methods=['POST'])
@cross_origin()
def index():
    data = request.json
    print(data)
    return jsonify([data['blurb'],data['blurb']])

@app.route('/pred', methods=['POST'])
def pred():
    data = request.json
    cat = find_cat(data.blurb)
    return json.dumps({ "cat":cat,"pichart":piechart(cat), "barchart":barchart(cat), "projects":top_proj(cat)})

if __name__ == '__main__':
    app.run(debug=True)