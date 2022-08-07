import time
import pathlib
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    # RocCurveDisplay,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV  # --> GridSearchCV trop lent
# from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV  # --> ne supporte pas le multi-scoring


def print_classification_report(y_true, y_pred):
    """Display a classification report based on the provided lists

    Parameters
    ----------
    y_true: list
        the expected values
    y_pred: list
        the predicted values
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Prediction = 0", "Prediction = 1"],
        zero_division=0,
    )
    print("--- Classification Report ---".ljust(100, "-"), "\n\n", report)


def print_confusion_matrix(y_true, y_pred):
    """Display a confusion matrix based on the provided lists

    Parameters
    ----------
    y_true: list
        the expected values
    y_pred: list
        the predicted values
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    print("--- Confusion Matrix ---".ljust(100, "-"), "\n")
    plt.show()


def print_rocauc(y_true_dict, y_pred_dict, figsize=[5, 5], ax=None, top_others=3):
    """Display the 'top_others' best ROC Curves + the last provided ROC Curve

    Parameters
    ----------
    y_true_dict: list
        the expected values for several models
    y_pred_dict: list
        the predicted values for several models
    """

    print("--- ROC AUC ---".ljust(100, "-"), "\n")
    auc_scores = {}
    # last_index = len(y_pred_dict)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt

    # find top scores:
    last_score_name = list(y_pred_dict)[-1]
    sorted_scores = defaultdict(list)
    for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        if model_name != last_score_name:
            y_true = y_true_dict[model_name]
            roc_score = roc_auc_score(y_true, y_pred)
            sorted_scores[model_name] = roc_score
    sorted_scores = sorted(sorted_scores, key=lambda x: sorted_scores[x], reverse=True)[
        :top_others
    ]
    sorted_scores.append(last_score_name)

    # display
    # for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
    for i, model_name in enumerate(sorted_scores):
        alpha_v = 1 if i == min(top_others, len(sorted_scores) - 1) else 0.2

        y_true = y_true_dict[model_name]
        y_pred = y_pred_dict[model_name]
        roc_score = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        ax.plot(fpr, tpr, label=f"{model_name} ({roc_score:.2f})", alpha=alpha_v)
        auc_scores[model_name] = roc_score

    ax.plot(
        [0, 1], [0, 1], label="Random (0.5)", linestyle="--", color="red", alpha=0.5
    )
    plt.xlabel("FPR (Positive label: 1)")
    plt.ylabel("TPR (Positive label: 1)")
    # plt.legend()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.show()

    return auc_scores


def print_prauc(y_true_dict, y_pred_dict, figsize=[5, 5], ax=None, top_others=3):
    """Display the 'top_others' best Precision Recall Curves + the last provided Precision Recall Curve

    Parameters
    ----------
    y_true_dict: list
        the expected values for several models
    y_pred_dict: list
        the predicted values for several models
    """

    print("--- PRECISION RECALL AUC ---".ljust(100, "-"), "\n")
    auc_scores = {}
    # last_index = len(y_pred_dict)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt

    # find top scores:
    last_score_name = list(y_pred_dict)[-1]
    sorted_scores = defaultdict(list)
    for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        if model_name != last_score_name:
            y_true = y_true_dict[model_name]
            pr_score = average_precision_score(y_true, y_pred)
            sorted_scores[model_name] = pr_score
    sorted_scores = sorted(sorted_scores, key=lambda x: sorted_scores[x], reverse=True)[
        :top_others
    ]
    sorted_scores.append(last_score_name)

    # display
    # for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
    for i, model_name in enumerate(sorted_scores):
        alpha_v = 1 if i == min(top_others, len(sorted_scores) - 1) else 0.2

        y_true = y_true_dict[model_name]
        y_pred = y_pred_dict[model_name]
        pr_score = average_precision_score(y_true, y_pred)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

        ax.plot(
            recall, precision, label=f"{model_name} ({pr_score:.2f})", alpha=alpha_v
        )
        auc_scores[model_name] = pr_score

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    ax.plot(
        [0, 1],
        [no_skill, no_skill],
        label="No skill",
        linestyle="--",
        color="red",
        alpha=0.3,
    )
    ax.plot([1, 0], [0, 1], label="Balanced", linestyle="--", color="green", alpha=0.5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.legend()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.show()

    return auc_scores


def save_score(
    method_name, threshold, param_grid, training_time, inference_time, **scores
):
    """Save the scores into the 'scores_df' DataFrame and to the 'scores_path' CSV file.
    Each call to this function appends exactly one row to the DataFrame and hence to the CSV.

    Parameters
    ----------
    method_name: str
        the name used to identify the record in the list
    threshold: float
        the threshold used to get the provided scores
    param_grid: dict
        the parameter grid used to get the provided scores
    training_time: float
        the time needed for the fitting process
    inference_time: float
        the time needed for the prediction process
    scores: list of parameters
        the scores to register
    """

    idx = np.where(scores_df.Method == method_name)[0]
    idx = idx[0] if idx.size > 0 else len(scores_df.index)

    rocauc_value = scores.get("roc_auc", None)
    f1_value = scores.get("f1", None)
    accuracy_value = scores.get("accuracy", None)
    precision_value = scores.get("precision", None)
    recall_value = scores.get("recall", None)
    prauc_value = scores.get("average_precision", None)
    TP = scores.get("TP", None)
    FP = scores.get("FP", None)
    TN = scores.get("TN", None)
    FN = scores.get("FN", None)

    scores_df.loc[idx] = [
        method_name,
        threshold,
        param_grid,
        rocauc_value,
        prauc_value,
        f1_value,
        accuracy_value,
        precision_value,
        recall_value,
        TP,
        TN,
        FP,
        FN,
        training_time,
        inference_time,
    ]
    scores_df.to_csv(scores_path, index=False)


def init_scores(file_path="data/scores.csv", append=False):
    global scores_df, scores_path, y_preds, y_trues

    scores_df = pd.DataFrame(
        columns=[
            "Method",
            "threshold",
            "params",
            "ROC AUC",
            "PR AUC",
            "F1 score",
            "Accuracy",
            "Precision",
            "Recall",
            "TP",
            "TN",
            "FP",
            "FN",
            "Training time",
            "Inference time",
        ]
    )
    y_preds = {}
    y_trues = {}
    scores_path = pathlib.Path(file_path)
    if append is True and scores_path.is_file():
        scores_df = pd.read_csv(scores_path)
    else:
        scores_df.to_csv(scores_path, index=False)


def get_scores(
    method_name,
    y_ref,
    X_ref=None,
    model=None,
    y_pred=None,
    y_pred_proba=None,
    param_grid=None,
    threshold=None,
    training_time=None,
    inference_time=None,
    register=False,
    simple=False,
    show_classification=True,
    show_confusion=True,
    show_roccurves=True,
    **scores,
):
    """Compute / Display / Save scores for the provided model

    More precisely, it compute the scores then call various function to display and save them.

    Parameters
    ----------
    method_name: str
        the name used to identify the record in the list
    model:
        the model that needs to be evaluated
    y_pred:
        the predicted values (only if the model is not provided)
    y_pred_proba:
        the predicted proba values (only if the model is not provided)
    X_ref: list of lists
        the X values used to get the predictions
    y_ref: list
        the expected values
    param_grid: dict
        the parameter grid used to get the provided scores
    training_time: float
        the time needed for the fitting process
    inference_time: float
        the time needed for the prediction process
    scores: list of parameters
        the scores to register

    Return
    ------
    dict:
        the dictionary of the computed scores
    """

    if model is not None:
        y_pred, y_pred_proba, inference_time = predict(model, X_ref, threshold)

    if y_pred is None or y_pred_proba is None:
        raise Exception("We either need the model with a X_ref to compute y_pred & y_pred_proba or directly the y_pred & y_pred_proba")

    try:

        cm = confusion_matrix(y_ref, y_pred, labels=[0, 1])

        scores = {
            "roc_auc": roc_auc_score(y_ref, y_pred_proba),
            "f1": f1_score(y_ref, y_pred),
            "accuracy": accuracy_score(y_ref, y_pred),
            "precision": precision_score(y_ref, y_pred, zero_division=0),
            "recall": recall_score(y_ref, y_pred),
            "average_precision": average_precision_score(y_ref, y_pred_proba),
            "TN": cm[0][0],
            "FP": cm[0][1],
            "FN": cm[1][0],
            "TP": cm[1][1],
        }

    except NameError:
        print("We either need a model or the y_pred & y_pred_proba variables")

    # Register score and replace if it already exists
    if register:
        save_score(
            method_name, threshold, param_grid, training_time, inference_time, **scores
        )

    # Basic report
    scores_str = ""
    for key in scores.keys():
        if type(scores[key]) == np.float64 and key not in ["TP", "TN", "FP", "FN"]:
            scores_str += f"{key.upper().rjust(20)} : {scores[key]:.4f}\n"
    scores_str += f"\n{'TRAINING-TIME'.rjust(20)} : {training_time:.4f}\n{'INFERENCE-TIME'.rjust(20)} : {inference_time:.4f}\n"

    print(
        "-" * 100,
        "These information are based on the best estimator of the above cross-validation".center(
            100,
        ),
        "-" * 100,
        sep="\n",
        end="\n\n",
    )
    print(f"--- {method_name} ---".ljust(100, "-"), "\n\n", scores_str, sep="")

    if simple:
        return

    # Classification report
    if show_classification:
        print_classification_report(y_ref, y_pred)

    # Confusion Matrix
    if show_confusion:
        print_confusion_matrix(y_ref, y_pred)

    # ROC AUC curves
    if show_roccurves:
        y_preds[method_name] = y_pred_proba
        y_trues[method_name] = y_ref
        print_rocauc(y_trues, y_preds)
        print_prauc(y_trues, y_preds)

    return scores


def predict(model, X_ref, threshold=None):
    """Convenience function that generalize the prediction process

    Parameters
    ----------
    model:
        the model that needs to make predictions
    X_ref: list of lists
        the X values used to get the predictions
    threshold: float (None)
        the threshold used to get the provided scores

    Returns
    -------
    list
        the binary predictions
    list
        the probabilities
    float
        the time needed for the prediction process
    """

    t0 = time.perf_counter()

    try:
        y_pred_proba = model.predict_proba(X_ref)[:, 1]
    except Exception:
        y_pred_proba = model.predict(X_ref)

    if threshold:
        y_pred = get_labels_from_threshold(y_pred_proba, threshold)
    else:
        y_pred = model.predict(X_ref)

    tt = time.perf_counter() - t0
    return y_pred, y_pred_proba, tt


def get_labels_from_threshold(y_proba, threshold):
    """Convenience function that quickly convert proabilities to binary results

    Parameters
    ----------
    y_proba: list
        the list of probabilities
    threshold: float (None)
        the threshold used to make the choices

    Returns
    -------
    list
        the binary predictions
    """

    return (y_proba >= threshold).astype("int")


def find_best_threshold(model, X_valid, y_valid, eval_function):
    """Find the threshold that maximize the provided scoring function

    Parameters
    ----------
    model:
        the model that needs to make predictions
    X_valid: list of lists
        the X values used to get the predictions
    y_valid: list
        the expected values
    eval_function: function
        the scoring method used to find the best threshold

    Returns
    -------
    float
        the best score found for the provided metric
    float
        the threshold matching the best metric's score
    """

    best_threshold = 0.0
    best_score = 0.0

    try:
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
    except Exception:
        y_pred_proba = model.predict(X_valid)

    for threshold in np.arange(0, 1, 0.001):

        y_pred_threshold = get_labels_from_threshold(y_pred_proba, threshold)

        score = eval_function(y_valid, y_pred_threshold)
        if score >= best_score:
            best_threshold = threshold
            best_score = score

    return best_score, best_threshold


def fit_model(
    model,
    X_ref,
    y_ref,
    param_grid={},
    scoring="roc_auc",
    cv=5,
    verbose=2,
    register=True,
):
    """Search the best hyper-parameters for the provided model

    Parameters
    ----------
    model:
        the model that needs to make predictions
    X_ref: list of lists
        the X values used to get the predictions
    y_ref: list
        the expected values
    param_grid: dict
        the parameter grid used to get the provided scores
    scoring: str
        the scoring method to use when evaluating the model in the Grid Search CV process
    cv: int / CrossValidation
        the number of cross validations to apply OR the instance of a CrossValidation instance
    verbose: int
        defines how much details are printed while training the model
        0 : nothing
        1 : K-fold scores + results for test set
        2 : K-fold scores + results for test & train sets

    Returns
    -------
    dict
        a dictionnary containing:
        - grid: the grid search instance
        - model: the grid search best estimator
        - training_time: the fitting time
        - inference_time: the prediction time
        - param_grid: the parameters used for the grid search
    """

    fit_time = time.perf_counter()
    grid_model = RandomizedSearchCV(
        model,
        param_grid,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        cv=cv,
        random_state=0,
        refit=scoring,
    )
    # grid_model = HalvingRandomSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, verbose=0, cv=cv, min_resources=500, random_state=0)
    # grid_model = HalvingGridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, verbose=0, cv=cv, min_resources=500, random_state=0)
    # grid_model = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, verbose=0, cv=cv, refit="roc_auc", return_train_score=True)
    grid_model.fit(X_ref, y_ref)
    fit_time = time.perf_counter() - fit_time

    results = grid_model.cv_results_
    n_splits = cv.n_splits if hasattr(cv, "n_splits") else cv
    sets_list = ["test"] if verbose < 3 else ["train", "test"]

    # Print K-fold scores
    if verbose > 1:
        for i in range(n_splits):
            print("".center(100, "-"))

            for sample in sets_list:
                scores_str = f"{scoring.upper()}: {results[f'split{i}_{sample}_score'].mean():.4f}"
                print(f"FOLD-{i+1} {sample.upper().rjust(6)} scores | {scores_str}")

    # Print overall scores
    if verbose > 0:

        for sample in sets_list:
            print(
                "\n",
                f" {sample.upper()}-CV-SPLIT MEAN SCORES ".center(100, "-"),
                sep="",
            )
            mean_str = f"{scoring.upper()}: {results[f'mean_{sample}_score'].mean():.4f} (std:{results[f'std_{sample}_score'].mean():.4f})"
            print(f"\n- {mean_str}")

        print("\n", "".center(100, "-"), sep="")

    inf_time = pd.Series(grid_model.cv_results_["mean_score_time"]).mean()

    return {
        "grid": grid_model,
        "model": grid_model.best_estimator_,
        "training_time": fit_time,
        "inference_time": inf_time,
        "param_grid": param_grid,
    }  # , **scores_args}
