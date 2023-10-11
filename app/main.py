import numpy
from atari import AtariAI, AtariGame

from net import NeuralNetwork


GAME_NAME = "ALE/SpaceInvaders-v5"


def main():
    game = AtariGame(GAME_NAME)
    learn_env = game.initialize_atari_env_game(render_mode=None)
    wrapped_env = game.get_wrapped_env()

    shape = learn_env.observation_space.shape
    data_size = numpy.prod(shape)
    input_size = shape[2]
    hidden_layer = 50
    output_size = learn_env.action_space.n
    save_dir = "./app/data/"
    save_file_name = GAME_NAME
    epochs = 200

    net = NeuralNetwork(
        input_size=input_size,
        hidden_layers=hidden_layer,
        output_size=output_size,
        data_size=data_size,
    )

    ai = AtariAI(save_dir, save_file_name=save_file_name, module=net)
    ai.train(epochs=epochs, env=wrapped_env)
    ai.load()


main()
