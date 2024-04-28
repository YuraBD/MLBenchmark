import numpy as np
import common.common as common
from sklearn.metrics import precision_score, recall_score, f1_score


class ModelEvaluator():
    def __init__(self, model_runner, data_path, model_type, use_softmax=False):
        self.model_runner = model_runner
        self.use_softmax = use_softmax
        self.model_type = model_type
        self.x_data, self.y_data = common.load_data(data_path)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def do_full_inference(self):
        #Warm-up
        for single_input in self.x_data[:10]:
            self.model_runner.do_inference(single_input)

        inf_times = []
        results = []
        for single_input in self.x_data:
            output, inf_time = self.model_runner.do_inference(single_input)
            if self.model_type == 'classification':
                if self.use_softmax:
                    output = output.astype(np.float32)
                    output = self.softmax(output)
                output = np.argmax(output)
            inf_times.append(inf_time)
            results.append(output)

        return results, inf_times

    def evaluate_model(self):
        results, inf_times = self.do_full_inference()

        metrics = {
            'accuracy': None,
            'aver_inf_time': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'mape': None,
            'mse': None
        }

        metrics['aver_inf_time'] = sum(inf_times) / len(self.y_data)
        if self.model_type == 'classification':
            corr_pred = 0
            for acc_out, exp_out in list(zip(results, self.y_data)):
                if acc_out == exp_out:
                    corr_pred += 1
            metrics['accuracy'] = (corr_pred / len(self.y_data)) * 100
            metrics['precision'] = precision_score(self.y_data, results, average='macro', zero_division=0)
            metrics['recall'] = recall_score(self.y_data, results, average='macro', zero_division=0)
            metrics['f1'] = f1_score(self.y_data, results, average='macro', zero_division=0)
        else:
            metrics['mape'] = np.mean(np.abs((self.y_data - results) / self.y_data)) * 100
            metrics['mse'] = np.mean((self.y_data - results) ** 2)

        return metrics
