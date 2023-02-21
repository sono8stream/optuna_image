import cv2
import optuna
import numpy as np


class Evaluator:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def tune(self):
        '''
            チューニングを実施
        '''
        study = optuna.create_study()
        study.optimize(self.do_trial(), n_trials=100)
        print(study.best_params)

        return study.best_params

    def do_trial(self):
        '''
            評価関数を定義
        '''
        def objective(trial):
            bottom = trial.suggest_int('bottom', 0, 255)
            top = trial.suggest_int('top', 0, 255)
            tar = self.apply(bottom, top)
            return self.evaluate2(tar)

        return objective

    def apply(self, bottom, top):
        '''
            画像処理を実装
        '''
        ret, upper = cv2.threshold(self.input, bottom, 255, cv2.THRESH_BINARY)
        ret, lower = cv2.threshold(self.input, top, 255, cv2.THRESH_BINARY_INV)
        return cv2.bitwise_and(upper, lower)

    def evaluate(self, target):
        '''
            精度評価。シンプルな正解率計算
        '''
        # 精度評価
        all = target.size
        score = all-np.count_nonzero(target == output)
        print(all, score)
        return score

    def evaluate2(self, target):
        '''
            F値
        '''
        tp = np.count_nonzero(np.logical_and(target, output))
        tn = np.count_nonzero(np.logical_and(
            np.logical_not(target), np.logical_not(output)))
        fp = np.count_nonzero(np.logical_and(target, np.logical_not(output)))
        fn = np.count_nonzero(np.logical_and(np.logical_not(target), output))
        print(tp, tn, fp, fn)
        precision = tp/(tp+fp) if tp+fp > 0 else 0
        recall = tp/(tp+fn) if tp+fn > 0 else 0
        f_measure = 2*precision*recall / \
            (precision+recall) if precision+recall > 0 else 0
        # precisionかf-measureが効果的
        return 1-precision


if __name__ == "__main__":
    # input = cv2.imread("data_sample/input.png")
    # output = cv2.imread("data_sample/output.png")
    input = cv2.imread("data_sample/grad2.png")
    output = cv2.imread("data_sample/grad2_out.png")
    evaluator = Evaluator(input, output)
    best_params = evaluator.tune()
    evaluator.evaluate(input)
    res = evaluator.apply(best_params['bottom'], best_params['top'])

    cv2.imshow("gt", output)
    cv2.imshow("estimation", res)
    cv2.imshow("mine", evaluator.apply(119, 129))
    # imx = evaluator.apply(119, 129)
    # evaluator.evaluate2(imx)
    cv2.waitKey(0)
