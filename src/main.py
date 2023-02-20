import cv2
import optuna
import numpy as np


class Evaluator:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def tune(self):
        study = optuna.create_study()
        study.optimize(self.do_trial(), n_trials=500)
        bottom = 0
        top = 255
        print(study.best_params)

        return study.best_params['bottom'], study.best_params['top']

    def do_trial(self):
        def objective(trial):
            bottom = trial.suggest_int('bottom', 0, 255)
            top = trial.suggest_int('top', 0, 255)
            tar = self.apply(bottom, top)
            return self.evaluate(tar)

        return objective

    def apply(self, bottom, top):
        ret, upper = cv2.threshold(self.input, bottom, 255, cv2.THRESH_BINARY)
        ret, lower = cv2.threshold(self.input, top, 255, cv2.THRESH_BINARY_INV)
        return cv2.bitwise_and(upper, lower)

    def evaluate(self, target):
        # 精度評価
        # print(target)
        # print(output)
        all = target.size  # チャネル数で割ってピクセル数を取得
        score = all-np.count_nonzero(target == output)
        print(all, score)
        return score


if __name__ == "__main__":
    input = cv2.imread("data/grad.png")
    output = cv2.imread("data/grad_out.png")
    evaluator = Evaluator(input, output)
    bottom, top = evaluator.tune()
    evaluator.evaluate(input)
    res = evaluator.apply(bottom, top)
    cv2.imshow("gt", output)
    cv2.imshow("estimation", res)
    cv2.imshow("mine", evaluator.apply(130, 140))
    cv2.waitKey(0)
