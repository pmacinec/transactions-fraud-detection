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

    df = pd.DataFrame(data, columns=['Generation', 'CV score (roc auc)', 'Metric'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(ylim=(0.55, 0.85))
    sns.lineplot(data=df, x='Generation', y='CV score (roc auc)', hue='Metric', ax=ax).set_title(title)

    plt.show()


def show_optimization_run_plot_for_multiple_tasks(tasks, title='', **kwargs):
    """
    Plot the run of multiple optimization tasks by generation.
    :param task: a tasks to be plotted
    :param metric: the metric to be plotted
    :param title: the title of the plot
    :return: None
    """

    metric = kwargs.get('metric', 'max')
    
    data = []
    
    for run, task in enumerate(tasks):
        # invert the scores because the library did the minimization task
        gen_scores_max = invert(task.gen_scores_min)
        gen_scores_min = invert(task.gen_scores_max)
        gen_scores_mean = invert(task.gen_scores_mean)

        for generation, value in enumerate(gen_scores_mean):
            data.append([run, generation, value, 'mean'])

        for generation, value in enumerate(gen_scores_max):
            data.append([run, generation, value, 'max'])

        for generation, value in enumerate(gen_scores_min):
            data.append([run, generation, value, 'min'])    
    
    df = pd.DataFrame(data, columns=['Run', 'Generation', 'CV score (roc auc)', 'Type'])
    df = df[df['Type'] == metric]

    for ylim in [(0.65, 0.8), None]:
        fig, ax = plt.subplots(figsize=(10, 8))
            
        if ylim is not None:
            ax.set(ylim=ylim)
            
        sns.lineplot(data=df[['Generation', 'CV score (roc auc)', 'Run']], x='Generation', y='CV score (roc auc)', hue='Run', ax=ax).set_title(title)
        plt.show()
