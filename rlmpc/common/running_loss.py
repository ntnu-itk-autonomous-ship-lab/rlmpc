"""
running_loss.py

Summary:
    Class for a running loss to be used in DNN training.

Author: Trym Tengesdal
"""


class RunningLoss:
    def __init__(self, batch_size: int) -> None:
        self.batch_size: int = batch_size
        self.aggregated_loss: float = 0.0
        self.n_samples: int = 0
        self.average_loss: float = 0.0
        self.reset()

    def update(self, loss: float) -> None:
        self.aggregated_loss += loss / self.batch_size
        self.n_samples += 1
        self.average_loss = self.aggregated_loss / (self.n_samples)

    def reset(self):
        self.aggregated_loss = 0.0
        self.n_samples = 0
        self.average_loss = 0.0
