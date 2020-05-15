from NiaPy.benchmarks import Benchmark


class ClassificationBenchmark(Benchmark):
    """
    NiaPy benchmark for classification task.

    :param model_fn: function which returns sklearn model.
    :param eval_fn: function(y_test, y_pred) which evaluates predictions
         and returns a scalar.
    :param x_train: train data.
    :param y_train: train labels.
    :param x_test: test data.
    :param y_test: test labels.
    """
    def __init__(self, model_fn, eval_fn, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_fn = model_fn
        self.eval_fn = eval_fn

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

    def function(self):
        def evaluate(_, solution_vec):
            selected_columns = self.select_columns(solution_vec)

            # fix of incorrect serialization when using multi threading module
            if len(selected_columns) == 1 and \
                    not isinstance(selected_columns[0], str):
                selected_columns = selected_columns[0]

            if len(selected_columns) < 1:
                # inverted score, since the optimizer minimizes the task
                return 1 - 0

            clf = self.model_fn()
            clf = clf.fit(self.x_train[selected_columns], self.y_train)

            y_pred = clf.predict(self.x_test[selected_columns])
            score = self.eval_fn(self.y_test, y_pred)

            # inverted score, since the optimizer minimizes the task
            return 1 - score

        return evaluate
