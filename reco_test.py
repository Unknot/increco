from river import optim
from river import reco
import csv


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


def main():
    model = reco.BiasedMF(
        n_factors=10,
        bias_optimizer=optim.SGD(0.025),
        latent_optimizer=optim.SGD(0.025),
        latent_initializer=optim.initializers.Normal(mu=0., sigma=0.1, seed=71)
    )

    with open("100k_a.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        for data_point in data:
            # print(data_point)
            example = ({
                "user": data_point[0],
                "item": data_point[1],
                },
                float(data_point[4]) - float(data_point[3])
                )
            # print(example)
            # return
            model.learn_one(**example[0], y=example[1])

    # for x, y in dataset:
    #     model.learn_one(**x, y=y)

    print(model.predict_one(user='Bob', item='Harry Potter'))


if __name__ == "__main__":
    main()
