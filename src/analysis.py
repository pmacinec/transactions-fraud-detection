import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def invert(scores):
    """
    Invert the scores.
    :param scores: the list of scores
    :return: inverted scores.
    """
    return list(map(lambda a: 1 - a, scores))


def show_optimization_run_plot(task, title=''):
    """
    Plot the run of the optimization task by generation.
    :param task: a task to be plotted
    :param title: the title of the plot
    :return: None
    """

    # invert the scores because the library did the minimization task
    gen_scores_max = invert(task.gen_scores_min)
    gen_scores_min = invert(task.gen_scores_max)
    gen_scores_mean = invert(task.gen_scores_mean)

    data = []

    for index, value in enumerate(gen_scores_mean):
        data.append([index, value, 'mean'])

    for index, value in enumerate(gen_scores_max):
        data.append([index, value, 'max'])

    for index, value in enumerate(gen_scores_min):
        data.append([index, value, 'min'])

    df = pd.DataFrame(data, columns=['generation', 'score', 'type'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(ylim=(0.5, 1))
    sns.lineplot(data=df, x='generation', y='score', hue='type', ax=ax).set_title(title)

    plt.show()

