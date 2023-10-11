import gymnasium as gym
from torch import nn
import torch
import numpy as np
import logging

logger = logging.getLogger("ai_game")


class AtariGame:
    """docstring for AtariGame
    Class to create environment of a atari game
    """

    def __init__(self, game: str) -> None:
        self.game = game
        self.env: gym.Env = None

    def initialize_atari_env_game(self, render_mode="rgb_array"):
        self.env = gym.make(self.game, render_mode=render_mode)
        return self.env

    def get_wrapped_env(self, deque_size=50) -> gym.wrappers.RecordEpisodeStatistics:
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(
            self.env, deque_size=deque_size
        )
        return wrapped_env


class AtariAI:
    def __init__(
        self,
        save_dir: str,
        save_file_name: str,
        module: nn.Module,
        data_size: int,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.MSELoss = None,
    ) -> None:
        self.save_dir = save_dir
        self.save_file_name = save_file_name
        self.save_path = self.save_dir + f"{self.save_file_name}.chkpt"
        self.module = module
        self.optimizer: torch.optim.Optimizer = (
            optimizer if optimizer else torch.optim.SGD(module.parameters(), lr=0.01)
        )
        self.criterion: torch.nn.MSELoss = (
            criterion if criterion else torch.nn.MSELoss()
        )
        self.data_size = data_size

    def train(self, epochs: int, env: gym.Env) -> None:
        self.module.train()

        for i in range(epochs):
            done = False
            state, info = env.reset()
            reward = 0
            score = 0
            while not done:
                state = torch.tensor(np.array([state]), dtype=torch.float32).view(
                    -1, self.data_size
                )
                state = state.add(score)
                self.optimizer.zero_grad()
                action_tensor = self.module(state.float(), model="online")
                action = torch.argmax(action_tensor).item()
                next_state, reward, n_state, done, info = env.step(action)

                target = self.module(state, model="target")
                loss = self.criterion(action_tensor, target)

                loss.backward()
                self.optimizer.step()
                state = next_state
                score += reward
                done = info.get("lives") == 0
            logger.debug(f" Episode: {i} Score: {score} ")
        self.save()
        env.close()

    def play(self, env: gym.Env) -> int:
        state, info = env.reset()
        done = False
        self.module.eval()
        score = 0
        while not done:
            env.render()
            state = torch.tensor(np.array([state]), dtype=torch.float32).view(
                -1, self.data_size
            )
            action_tensor = self.module(state)
            action = torch.argmax(action_tensor).item()
            state, reward, n_state, done, info = env.step(action)
            done = info.get("lives") == 0
            score += reward
        env.close()
        return info

    def load(self) -> None:
        try:
            state_dict = torch.load(self.save_path)
            self.module.load_state_dict(state_dict)
        except Exception:
            logger.debug("Nao há arquivo para carregar")

    def save(self) -> None:
        torch.save(
            self.module.state_dict(),
            self.save_path,
        )
