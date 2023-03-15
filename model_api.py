from river import optim
from river import reco
from river import metrics
from tqdm import tqdm
import pickle
import csv
from flask import Flask, request
import json


class BiasedRecoModel:
    def __init__(self, n_factors=5):
        self.model = reco.BiasedMF(
            n_factors=n_factors,
            bias_optimizer=optim.Nadam(),
            latent_optimizer=optim.Nadam(),
            latent_initializer=optim.initializers.Normal(mu=0., sigma=0.1)
        )
        self.metric = metrics.MAE()
        self.is_trained = False

    def train(self, data_set):
        for data_point in tqdm(data_set):
            x, y = data_point
            self.model.learn_one(**x, y=y)
        self.is_trained = True

    def save(self):
        with open("./.models/model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open("./.models/model.pkl", "rb") as f:
            self.model = pickle.load(f)


def load_training_data():
    with open("./.data/steam-200k.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        training_data = [(
            {"user": data_point[0],
             "item": data_point[1]},
            float(data_point[3]))
            for data_point in data if data_point[2] == "purchase"]
    return training_data


global model

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "IncReco model"


@app.route("/train_model", methods=["GET"])
def train_model():
    print("Model training...")
    model = BiasedRecoModel(10)
    training_data = load_training_data()
    model.train(training_data)
    model.save()
    return "Model trained and saved."


@app.route("/load_model", methods=["GET"])
def load_model():
    print("Loading model...")
    global model
    model = BiasedRecoModel(10)
    model.load()
    return "Model loaded"


@app.route("/predict", methods=["POST"])
def predict():
    global model
    # if not model.is_trained:
    #     return "Model not trained. Train the model first or load a trained model."
    print(model.is_trained)
    prediction = model.model.predict_one(**dict(request.json))
    return f"Predicted score: {prediction}"


if __name__ == "__main__":
    # m = BiasedRecoModel(10)
    # training_data = load_training_data()
    # m.train(training_data)
    # m.save()
    app.run(debug=True, host='0.0.0.0')
