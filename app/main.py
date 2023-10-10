import torch
from atari import AtariAI, AtariGame

from net import NeuralNetwork


GAME_NAME = "ALE/SpaceInvaders-v5"


def main():
    game = AtariGame(GAME_NAME)
    learn_env = game.initialize_atari_env_game(render_mode=None)

    input_size = learn_env.observation_space.shape
    input_size = input_size[0]
    hidden_layer = 2
    output_size = learn_env.action_space.n
    save_dir = "./data/"
    save_file_name = GAME_NAME
    epochs = 200

    net = NeuralNetwork(
        input_size=input_size, hidden_layers=hidden_layer, output_size=output_size
    )

    ai = AtariAI(save_dir, save_file_name=save_file_name, module=net)
    ai.train(epochs=epochs, env=learn_env)
    ai.load()


main()
