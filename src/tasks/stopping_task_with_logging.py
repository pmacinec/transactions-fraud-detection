import numpy as np

from NiaPy.task import StoppingTask


class StoppingTaskWithLogging(StoppingTask):
    def __init__(self, **kwargs):
        StoppingTask.__init__(self, **kwargs)

        self.gen_scores_mean = []
        self.gen_scores_max = []
        self.gen_scores_min = []

        self.__current_gen_scores = []

    def eval(self, A):
        x_f = super().eval(A)
        self.__current_gen_scores.append(x_f)
        return x_f

    def nextIter(self):
        super().nextIter()

        if len(self.__current_gen_scores):
            self.gen_scores_mean.append(np.mean(self.__current_gen_scores))
            self.gen_scores_max.append(np.max(self.__current_gen_scores))
            self.gen_scores_min.append(np.min(self.__current_gen_scores))

        self.__current_gen_scores = []
