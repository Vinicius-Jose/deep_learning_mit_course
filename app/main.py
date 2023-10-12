import numpy
from atari import AtariAI, AtariGame

from net import NeuralNetwork
import logging
import torch

torch.manual_seed(1)

logger = logging.getLogger("ai_game")
logger.setLevel(logging.DEBUG)


GAME_NAME = "ALE/SpaceInvaders-v5"
SAVE_DIR = "./app/data/"


def load_game(render_mode: str, hidden_layer: int = 50) -> AtariAI:
    game = AtariGame(GAME_NAME)
    env = game.initialize_atari_env_game(render_mode=render_mode)
    wrapped_env = game.get_wrapped_env()

    shape = env.observation_space.shape
    input_size = numpy.prod(shape)

    output_size = env.action_space.n

    save_file_name = GAME_NAME

    net = NeuralNetwork(
        input_size=input_size,
        hidden_layers=hidden_layer,
        output_size=output_size,
    )

    ai = AtariAI(
        SAVE_DIR, save_file_name=save_file_name, module=net, data_size=input_size
    )
    return ai, wrapped_env


def main():
    epochs = 80
    ai, wrapped_env = load_game(render_mode=None)
    ai.load()
    ai.train(epochs=epochs, env=wrapped_env)

    ai, wrapped_env = load_game(render_mode="human")
    ai.load()
    info = ai.play(wrapped_env)

    logger.debug(info)


main()
