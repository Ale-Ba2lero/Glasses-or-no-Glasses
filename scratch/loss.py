
import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses, acc = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.backward(output, y)
        return data_loss, acc
    
class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip the values to avoid dividing by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihood = -np.log(correct_confidences)
        acc = sum([1 for i in range(samples) if np.argmax(y_pred[i]) == y_true[i]])/samples

        return negative_log_likelihood, acc

    def backward(self, y_pred, y_true):
        num_examples = len(y_pred)
        self.doutput = y_pred
        self.doutput[range(num_examples), y_true] -= 1
        self.doutput /= num_examples
