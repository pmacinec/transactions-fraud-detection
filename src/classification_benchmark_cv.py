from NiaPy.benchmarks import Benchmark
from sklearn.model_selection import KFold
from functools import reduce

import numpy as np


class ClassificationBenchmarkCV(Benchmark):
    """
    NiaPy benchmark for classification task with cross-validation.

    :param model_fn: function which returns sklearn model.
    :param eval_fn: function(y_test, y_pred) which evaluates predictions
         and returns a scalar.
    :param x_train: train data.
    :param y_train: train labels.
    :param x_test: test data.
    :param y_test: test labels.
    :param cv: number of cross-validation folds.
    :param random_state: random_state for cross-validation.
    """
    def __init__(self, model_fn, eval_fn, x_train, y_train, cv=3, random_state=None):
        self.x_train = x_train
        self.y_train = y_train
        self.model_fn = model_fn
        self.eval_fn = eval_fn
        self.k_fold = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        
        self.cache = {}

        Benchmark.__init__(self, 0, 1)

    def get_length(self):
        """
        Get length of the vector which is being optimized.

        :return: length of the vector which is being optimized.
        """
        return len(self.x_train.columns)

    def select_columns(self, solution_vec):
        """
        Select columns based on the solution vector.

        :param solution_vec: solution of the problem as a vector.
        :return: list of column names based on the solution vector.
        """
        return self.x_train.columns[solution_vec >= 0.5].tolist()
    
    def get_solution_vec_key(self, solution_vec):
        """
        Get cache key from the solution vector.

        :param solution_vec: solution of the problem as a vector.
        :return: cache key string.
        """
        return int(reduce(lambda a, b: f"{a}{int(b)}", solution_vec >= 0.5, ""), 2)    

    def function(self):
        def evaluate(_, solution_vec):
            solution_vec_key = self.get_solution_vec_key(solution_vec)
            
            if solution_vec_key in self.cache:
                return self.cache[solution_vec_key]
            
            selected_columns = self.select_columns(solution_vec)

            # fix of incorrect serialization when using multi threading module
            if len(selected_columns) == 1 and \
                    not isinstance(selected_columns[0], str):
                selected_columns = selected_columns[0]

            if len(selected_columns) < 1:
                # inverted score, since the optimizer minimizes the task
                score = 1 - 0
                self.cache[solution_vec_key] = score                               
                
                return score
            
            scores = []
            
            for train_index, test_index in self.k_fold.split(self.x_train):
                X_train, X_test = self.x_train.iloc[train_index], self.x_train.iloc[test_index]
                y_train, y_test = np.asarray(np.array(self.y_train)[train_index]), np.asarray(np.array(self.y_train)[test_index])
                
                clf = self.model_fn()
                clf = clf.fit(X_train[selected_columns], y_train)

                y_pred = clf.predict(X_test[selected_columns])
                score = self.eval_fn(y_test, y_pred)
                
                scores.append(score)

            # inverted score, since the optimizer minimizes the task
            score = 1 - np.mean(scores)
            self.cache[solution_vec_key] = score

            return score

        return evaluate
