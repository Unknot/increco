from river import optim
from river import reco


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

    for x, y in dataset:
        model.learn_one(**x, y=y)

    print(model.predict_one(user='Bob', item='Harry Potter'))


if __name__ == "__main__":
    main()
