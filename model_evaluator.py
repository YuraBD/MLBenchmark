import numpy as np
import common.common as common


class ModelEvaluator():
    def __init__(self, model_runner, data_path, model_type, use_softmax=False):
        self.model_runner = model_runner
        self.use_softmax = use_softmax
        self.model_type = model_type
        self.x_data, self.y_data = common.load_data(data_path)

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def do_full_inference(self):
        #Warm-up
        for input in self.x_data[:10]:
            self.model_runner.do_inference(input)

        inf_times = []
        results = []
        for input in self.x_data:
            output, inf_time = self.model_runner.do_inference(input)
            if self.use_softmax:
                output = self.softmax(output)
            if self.model_type == 'classification':
                output = np.argmax(output)
            inf_times.append(inf_time)
            results.append(output)

        return results, inf_times

    def evaluate_model(self):
        results, inf_times = self.do_full_inference()

        corr_pred = 0
        for acc_out, exp_out in list(zip(results, self.y_data)):
            if acc_out == exp_out:
                corr_pred += 1

        accuracy = (corr_pred / len(self.y_data)) * 100
        aver_inf_time = sum(inf_times) / len(self.y_data)

        return accuracy, aver_inf_time