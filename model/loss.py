import numpy as np
import abc


class Loss(abc.ABC):

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    @abc.abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> (np.ndarray, float):
        pass

    @abc.abstractmethod
    def backpropagation(self) -> np.ndarray:
        pass

    def calculate(self, output: np.ndarray, y: np.ndarray):
        sample_losses, acc = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        d_score = self.backpropagation()
        return data_loss, acc, d_score


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> (float, float):
        self.y_pred: np.ndarray = y_pred
        self.y_true: np.ndarray = y_true

        samples = len(self.y_pred)
        # clip the values to avoid dividing by zero
        y_pred_clipped = np.clip(self.y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihood = -np.log(correct_confidences)

        # calculate accuracy
        acc = sum([1 for i in range(samples) if np.argmax(self.y_pred[i]) == self.y_true[i]]) / samples
        return negative_log_likelihood, acc

    def backpropagation(self):
        num_examples = len(self.y_pred)
        d_score = self.y_pred
        d_score[range(num_examples), self.y_true] -= 1
        d_score /= num_examples
        return d_score
