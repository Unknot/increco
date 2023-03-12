from river import optim
from river import reco
from river import metrics
import csv
import ast
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")


dataset = (
    ({'user': 'Alice', 'item': 'Superman'}, 8),
    ({'user': 'Alice', 'item': 'Terminator'}, 9),
    ({'user': 'Alice', 'item': 'Star Wars'}, 8),
    ({'user': 'Alice', 'item': 'Notting Hill'}, 2),
    ({'user': 'Alice', 'item': 'Harry Potter'}, 5),
    ({'user': 'Bob', 'item': 'Superman'}, 8),
    ({'user': 'Bob', 'item': 'Terminator'}, 9),
    ({'user': 'Bob', 'item': 'Star Wars'}, 8),
    ({'user': 'Bob', 'item': 'Notting Hill'}, 2)
)

filter_problematic = False


def twitch():
    model = reco.BiasedMF(
        n_factors=10,
        bias_optimizer=optim.Adam(),
        latent_optimizer=optim.Adam(),
        latent_initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=71)
    )

    metric = metrics.MAE()

    problematic_users = set()
    problematic_items = set()

    if filter_problematic:
        with open("./.data/problematic_users.txt", "r") as f:
            ser = f.read()
            problematic_users = \
                set() if ser == str(set()) \
                else ast.literal_eval(ser)
            print(f"Problematic users recovered: {len(problematic_users)}")
        with open("./.data/problematic_items.txt", "r") as f:
            ser = f.read()
            problematic_items = \
                set() if ser == str(set()) \
                else ast.literal_eval(ser)
            print(f"Problematic items recoveres: {len(problematic_items)}")

    # Working with the 100k Twitch data from here:
    # https://cseweb.ucsd.edu/~jmcauley/datasets.html
    with open("./.data/100k_a.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        for data_point in data:
            # print(data_point)
            example = (
                {
                    "user": data_point[0],
                    "item": data_point[2],
                },
                float(data_point[4]) - float(data_point[3])
            )
            # print(example)
            # return
            # if example[1] < 1. or example[1] > 50:
            if example[0]["user"] in problematic_users or \
                    example[0]["item"] in problematic_items:
                print("Problematic, skipping example:")
                print(example)
                print(metric)
                continue

            try:
                y_pred = model.predict_one(**example[0])
                model.learn_one(**example[0], y=example[1])
                metric.update(example[1], y_pred)
                # print(metric)
            except RuntimeWarning:
                print(example)
                problematic_users |= set([example[0]["user"]])
                problematic_items |= set([example[0]["item"]])

    # for x, y in dataset:
    #     model.learn_one(**x, y=y)

    print(metric)

    print(model.predict_one(user='Bob', item='Harry Potter'))
    print(model.predict_one(user='Putin', item='Brokeback Mountain'))

    print(f"Problematic users found: {len(problematic_users)}")
    print(f"Problematic items found: {len(problematic_items)}")

    with open("problematic_users.txt", "w") as f:
        f.write(str(problematic_users))
    with open("problematic_items.txt", "w") as f:
        f.write(str(problematic_items))


def steam():
    model = reco.BiasedMF(
        n_factors=5,
        bias_optimizer=optim.Nadam(),
        latent_optimizer=optim.Nadam(),
        latent_initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=71)
    )

    metric = metrics.MAE()
    maes = []

    with open("./.data/steam-200k.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        for idx, data_point in enumerate(data):
            if data_point[2] == "purchase":
                continue
            example = (
                {
                    "user": data_point[0],
                    "item": data_point[1],
                },
                float(data_point[3])
            )

            # if idx == 2390:
            #     print(example)
            #     print(metric)
            #     print(maes)
            #     return

            # print(example)
            # return
            try:
                if idx > 0:
                    y_pred = model.predict_one(**example[0])
                model.learn_one(**example[0], y=example[1])
                if idx > 0:
                    metric.update(example[1], y_pred)
                maes.append(metric.get())
                # print(metric)
            except RuntimeWarning:
                print(example)

            if math.isnan(metric.get()) or math.isinf(metric.get()):
                print(idx-1)
                break

            # if idx > 2388:
            #     break

    print(metric.get())
    print(math.isnan(metric.get()))
    plt.plot(maes)
    plt.show()


if __name__ == "__main__":
    twitch()
