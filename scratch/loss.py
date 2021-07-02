import numpy as np
import abc


class Loss(abc.ABC):

    @abc.abstractmethod
    def forward(self, y_pred: np.array, y_true: np.array) -> (np.array, float):
        pass

    @abc.abstractmethod
    def backward(self, y_pred: np.array, y_true: np.array) -> np.array:
        pass

    def calculate(self, output, y):
        sample_losses, acc = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        d_score = self.backward(output, y)
        return data_loss, acc, d_score


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip the values to avoid dividing by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihood = -np.log(correct_confidences)
        acc = sum([1 for i in range(samples) if np.argmax(y_pred[i]) == y_true[i]]) / samples

        return negative_log_likelihood, acc

    def backward(self, y_pred, y_true):
        num_examples = len(y_pred)
        d_score = y_pred
        d_score[range(num_examples), y_true] -= 1
        d_score /= num_examples

        return d_score
