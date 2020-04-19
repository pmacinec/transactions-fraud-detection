from sklearn.metrics import classification_report, plot_confusion_matrix,\
    roc_auc_score


def custom_classification_report(clf, x_test, y_test, **kwargs):
    """
    Create custom classification report.

    :param clf: classifier model.
    :param x_test: test samples to predict labels for.
    :param y_test: true label values of test samples.
    :param confussion_matrix: wether to plut confussion matrix.
    """
    y_pred = clf.predict(x_test)
    confussion_matrix = kwargs.get('confussion_matrix', True)

    clf_report = classification_report(
        y_pred,
        y_test,
        target_names=['not fraud', 'is fraud'],
        output_dict=True
    )

    # Custom print because of incorrect formatting of original function
    for key in clf_report:
        if isinstance(clf_report[key], dict):
            print(f'\033[1m{key}\033[0m')

            for metric in clf_report[key]:
                print(f'{metric}: {clf_report[key][metric]}')
        else:
            print(f'{key}: {clf_report[key]}')

        print('\n')

    print(f'\033[1mArea Under the Receiver Operating Characteristic Curve '
          f'(ROC AUC)\033[0m')
    print(f'Score: {roc_auc_score(y_test, y_pred)}')
    print('\n')

    if confussion_matrix:
        plot_confusion_matrix(clf, x_test, y_test)
