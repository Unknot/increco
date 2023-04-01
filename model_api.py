from river import optim
from river import reco
from river import metrics
from tqdm import tqdm
import pickle
import csv
import pandas as pd
from flask import Flask, request
from sklearn.model_selection import train_test_split


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
            pickle.dump(self, f)

    @staticmethod
    def load():
        with open("./.models/model.pkl", "rb") as f:
            model = pickle.load(f)
        return model

    def evaluate(self, data_set):
        metric = metrics.MAE()
        for x, y_true in data_set:
            y_pred = self.model.predict_one(**x)
            metric.update(y_true, y_pred)
        return metric


def process_steam_data():
    with open("./.data/steam-200k.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        parsed_data = [(
            {"user": data_point[0],
             "item": data_point[1]},
            float(data_point[3]))
            for data_point in data if data_point[2] == "purchase"]

    agg_data = pd.DataFrame.from_records([x[0] for x in parsed_data]) \
        .groupby("user") \
        .agg({"item": len}) \
        .reset_index()

    train_users, test_users, _, _ = train_test_split(agg_data["user"],
                                                     agg_data["item"],
                                                     test_size=0.3)
    # print(train_users)
    train_data = [data for data in parsed_data
                  if data[0]["user"] in list(train_users)]
    test_data = [data for data in parsed_data
                 if data[0]["user"] in list(test_users)]
    print(f"Len train_data:  {len(train_data)}")
    print(f"Len test_data:  {len(test_data)}")

    with open("./.data/steam_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("./.data/steam_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    return train_data, test_data


def load_steam_data():
    with open("./.data/steam_train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("./.data/steam_test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


global model

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "IncReco model"


@app.route("/train_model", methods=["GET"])
def train_model():
    print("Model training...")
    model = BiasedRecoModel(10)
    training_data, _ = load_steam_data()
    model.train(training_data)
    model.save()
    return "Model trained and saved."


@app.route("/load_model", methods=["GET"])
def load_model():
    print("Loading model...")
    global model
    model = BiasedRecoModel.load()
    return "Model loaded"


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if not model.is_trained:
        return """Model not trained. Train the model first
                 or load a trained model."""
    # print(model.is_trained)
    prediction = model.model.predict_one(**dict(request.json))
    return f"Predicted score: {prediction}"


@app.route("/evaluate", methods=["GET"])
def evaluate():
    global model
    if not model.is_trained:
        return """Model not trained. Train the model first
                 or load a trained model."""
    _, test_data = load_steam_data()
    print(set(x[0]["user"] for i, x in enumerate(test_data) if i < 1000))
    metric = model.evaluate(test_data)
    return str(metric)


@app.route("/rank", methods=["POST"])
def rank():
    global model
    if not model.is_trained:
        return """Model not trained. Train the model first
                 or load a trained model."""
    result = model.model.rank(**dict(request.json))
    return f"Rank: {result}"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
