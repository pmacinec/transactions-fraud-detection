from NiaPy.benchmarks import Benchmark


class ClassificationBenchmark(Benchmark):
    """
    NiaPy benchmark for classification task

    :param model_fn: function which returns sklearn model
    :param eval_fn: function(y_test, y_pred) which evaluates predictions and returns a scalar
    :param x_train: train data
    :param y_train: train labels
    :param x_test: test data
    :param y_test: test labels
    """
    def __init__(self, model_fn, eval_fn, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_fn = model_fn
        self.eval_fn = eval_fn

        self.columns = x_train.columns

        Benchmark.__init__(self, 0, 1)

    def get_length(self):
        """
        Get length of the vector which is being optimized.
        :return: length of the vector which is being optimized
        """
        return len(self.columns)

    def select_columns(self, solution_vec):
        """
        Select columns based on the solution vector.
        :param solution_vec: current solution of the problem as a vector
        :return: list of column names based on the solution vector
        """
        return list(self.columns[solution_vec > 0.5])

    def function(self):
        def evaluate(_, solution_vec):
            selected_columns = self.select_columns(solution_vec)

            if len(selected_columns) < 1:
                # return the inverted score, since the optimizer minimizes the task
                return 1 - 1

            clf = self.model_fn()
            clf = clf.fit(self.x_train[selected_columns], self.y_train)

            y_pred = clf.predict(self.x_test[selected_columns])
            score = self.eval_fn(self.y_test, y_pred)

            # print(len(selected_columns), score)

            # return the inverted score, since the optimizer minimizes the task
            return 1 - score

        return evaluate
