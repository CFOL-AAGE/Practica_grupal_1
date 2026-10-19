"""
Training of the models defined in models.py.

Sparse features (families 1 and 2): tune_sparse, train_sparse_model, train_all_sparse

Common to every family (they only need `id` and `label`):
    add_stratified_folds, best_params, save_train_info

Dense features (families 3 and 4): tune_dense, train_dense_model, train_all_dense
"""

import json
import time
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import CHAR_PATH, LEXICAL_PATH, LM_PATH, MODELS_DIR, SEED, STYLE_PATH, TUNING_FRACTION
from load_datasets import load_split
from models import build_dense_grid, build_dense_pipeline, build_sparse_grid, build_sparse_pipeline

# Parquet where each sparse column (families 1 and 2) lives
FEATURE_PATHS = {"tf": LEXICAL_PATH, "tf_nostop": LEXICAL_PATH, "tf_char": CHAR_PATH}
# Parquet where each dense column (families 3 and 4) lives
DENSE_PATHS = {"features_style": STYLE_PATH, "features_lm": LM_PATH}


def add_stratified_folds(df: DataFrame, num_folds: int) -> DataFrame:
    """
    Adds the column `fold` (0 .. num_folds - 1) with the same share of each class
    in every fold. Valid for any family (only uses `id` and `label`).

    Args:
        df: DataFrame
            Needs the columns `id` and `label`
        num_folds: int

    Returns:
        DataFrame
            df + `fold` (int)
    """
    # SEED + 2: independent from the hashes of assign_split (SEED and SEED + 1)
    by_class = Window.partitionBy("label").orderBy(F.xxhash64("id", F.lit(SEED + 2)))
    folds = (
        df.select("id", "label").
            withColumn("fold", ((F.row_number().over(by_class) - 1) % num_folds).cast("int")).
            select("id", "fold")
    )
    return df.join(F.broadcast(folds), on="id", how="inner")



def tune_sparse(train: DataFrame, input_col: str, model: str,
                num_folds: int = 3, parallelism: int = 2) -> CrossValidatorModel:
    """
    Selects the hyperparameters of one sparse model (families 1 and 2) with a
    stratified k-fold CrossValidator over a stratified sample of train
    (TUNING_FRACTION of each class).

    Args:
        train: DataFrame
            Output of load_split(..., "train")
        input_col: str
            "tf", "tf_nostop" or "tf_char"
        model: str
            "nb", "lr" or "svc"
        num_folds: int
            Number of folds of the cross validation
        parallelism: int
            Number of combinations trained at the same time

    Returns:
        CrossValidatorModel
            avgMetrics: mean AUC-PR of each combination over the folds
    """
    pipeline = build_sparse_pipeline(input_col, model)

    # Same fraction for each class: the sample keeps the 29 % of AI of train
    fractions = {0.0: TUNING_FRACTION, 1.0: TUNING_FRACTION}
    sample = train.sampleBy("label", fractions=fractions, seed=SEED)
    sample = add_stratified_folds(sample, num_folds)

    # AUC-PR of the class 1.0 (AI), over element [1] of the score column. In NB rawPrediction
    # is the log-likelihood of each class (it is not ordered by P(AI)): its probability is used
    score_col = "probability" if model == "nb" else "rawPrediction"
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol=score_col,
                                              metricName="areaUnderPR")

    # foldCol: the folds are the ones of add_stratified_folds instead of a random split
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=build_sparse_grid(pipeline, model),
                        evaluator=evaluator, numFolds=num_folds, foldCol="fold",
                        parallelism=parallelism)
    return cv.fit(sample)



def best_params(cv_model: CrossValidatorModel) -> tuple:
    """
    Returns the combination of hyperparameters with the best mean AUC-PR.
    Valid for the CrossValidatorModel of any family.

    Args:
        cv_model: CrossValidatorModel
            Output of tune_sparse (or of the CrossValidator of any family)

    Returns:
        tuple
            (ParamMap of the best combination, its mean AUC-PR)
    """
    metrics = cv_model.avgMetrics
    best = max(range(len(metrics)), key=metrics.__getitem__)
    return cv_model.getEstimatorParamMaps()[best], metrics[best]


def save_train_info(path: Path, params: dict, auc_pr: float, tune_s: float, fit_s: float) -> None:
    """
    Saves next to a trained model what evaluate.py reports about its training:
    the CV AUC-PR, the chosen hyperparameters and the times.

    Args:
        path: Path
            Folder of the saved PipelineModel
        params: dict
            ParamMap of the best combination (output of best_params)
        auc_pr: float
            Its mean AUC-PR over the folds
        tune_s: float
            Seconds of the CrossValidator
        fit_s: float
            Seconds of the final fit with the whole train
    """
    info = {"auc_pr_cv": auc_pr, "tune_s": tune_s, "fit_s": fit_s,
            "params": {p.name: v for p, v in params.items()}}
    (path / "train_info.json").write_text(json.dumps(info, indent=2))



def train_sparse_model(spark: SparkSession, input_col: str, model: str,
                       parallelism: int = 2) -> None:
    """
    Tunes one sparse model (families 1 and 2), fits it with the whole train and
    saves it. Skipped if the model was already saved.

    Args:
        spark: SparkSession
        input_col: str
            "tf", "tf_nostop" or "tf_char"
        model: str
            "nb", "lr" or "svc"
        parallelism: int
            Number of combinations trained at the same time in the CrossValidator
    """
    path = MODELS_DIR / f"{input_col}_{model}"
    if (path / "_SUCCESS").exists():        # written at the end of a complete save
        return

    # Only the column used: the other vectors of the parquet are not read (column pruning)
    train = load_split(spark, FEATURE_PATHS[input_col], "train").select("id", input_col, "label")

    start = time.perf_counter()
    cv_model = tune_sparse(train, input_col, model, parallelism=parallelism)
    tune_s = time.perf_counter() - start
    params, auc_pr = best_params(cv_model)
    print(f"[{path.name}] AUC-PR (CV) = {auc_pr:.4f} | "
          + ", ".join(f"{p.name}={v}" for p, v in params.items()))

    # bestModel was fitted only with the sample: refit the same Pipeline with the whole train
    start = time.perf_counter()
    final_model: PipelineModel = cv_model.getEstimator().fit(train, params)
    fit_s = time.perf_counter() - start

    final_model.write().overwrite().save(str(path))
    save_train_info(path, params, auc_pr, tune_s, fit_s)
    (path / "_SUCCESS").touch()


def train_all_sparse(spark: SparkSession, input_cols: tuple = ("tf", "tf_nostop", "tf_char"),
                     models: tuple = ("nb", "svc", "lr"), parallelism: int = 2) -> None:
    """
    Trains every combination of sparse column x model of the families 1 and 2
    (the saved ones are skipped).

    Args:
        spark: SparkSession
        input_cols: tuple
            Sparse columns to use
        models: tuple
            Models to train
        parallelism: int
            Number of combinations trained at the same time in the CrossValidator
    """
    for input_col in input_cols:
        for model in models:
            train_sparse_model(spark, input_col, model, parallelism)


def tune_dense(train: DataFrame, input_col: str, model: str,
               num_folds: int = 3, parallelism: int = 2) -> CrossValidatorModel:
    """
    Selects the hyperparameters of one dense model (families 3 and 4) with a
    stratified k-fold CrossValidator over a stratified sample of train
    (TUNING_FRACTION of each class).

    Args:
        train: DataFrame
            Output of load_split(..., "train")
        input_col: str
            Dense features column: "features_style" or "features_lm"
        model: str
            "lr" or "gbt"
        num_folds: int
            Number of folds of the cross validation
        parallelism: int
            Number of combinations trained at the same time

    Returns:
        CrossValidatorModel
            avgMetrics: mean AUC-PR of each combination over the folds
    """
    pipeline = build_dense_pipeline(input_col, model)

    # Same fraction for each class: the sample keeps the 29 % of AI of train
    fractions = {0.0: TUNING_FRACTION, 1.0: TUNING_FRACTION}
    sample = train.sampleBy("label", fractions=fractions, seed=SEED)
    sample = add_stratified_folds(sample, num_folds)

    # AUC-PR of the class 1.0 (AI); rawPrediction exists in LR and GBT
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                              metricName="areaUnderPR")

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=build_dense_grid(pipeline, model),
                        evaluator=evaluator, numFolds=num_folds, foldCol="fold",
                        parallelism=parallelism)
    return cv.fit(sample)


def train_dense_model(spark: SparkSession, input_col: str, model: str,
                      parallelism: int = 2) -> None:
    """
    Tunes one dense model (families 3 and 4), fits it with the whole train and
    saves it. Skipped if the model was already saved.

    Args:
        spark: SparkSession
        input_col: str
            "features_style" or "features_lm"
        model: str
            "lr" or "gbt"
        parallelism: int
            Number of combinations trained at the same time in the CrossValidator
    """
    path = MODELS_DIR / f"{input_col}_{model}"
    if (path / "_SUCCESS").exists():        # written at the end of a complete save
        return

    # Only the column used (column pruning)
    train = load_split(spark, DENSE_PATHS[input_col], "train").select("id", input_col, "label")

    start = time.perf_counter()
    cv_model = tune_dense(train, input_col, model, parallelism=parallelism)
    tune_s = time.perf_counter() - start
    params, auc_pr = best_params(cv_model)
    print(f"[{path.name}] AUC-PR (CV) = {auc_pr:.4f} | "
          + ", ".join(f"{p.name}={v}" for p, v in params.items()))

    # bestModel was fitted only with the sample: refit the same Pipeline with the whole train
    start = time.perf_counter()
    final_model: PipelineModel = cv_model.getEstimator().fit(train, params)
    fit_s = time.perf_counter() - start

    final_model.write().overwrite().save(str(path))
    save_train_info(path, params, auc_pr, tune_s, fit_s)
    (path / "_SUCCESS").touch()



def train_all_dense(spark: SparkSession, input_cols: tuple = ("features_style", "features_lm"),
                    models: tuple = ("lr", "gbt"), parallelism: int = 2) -> None:
    """
    Trains every combination of dense column x model of the families 3 and 4
    (the saved ones are skipped).

    Args:
        spark: SparkSession
        input_cols: tuple
            Dense columns to use
        models: tuple
            Models to train
        parallelism: int
            Number of combinations trained at the same time in the CrossValidator
    """
    for input_col in input_cols:
        for model in models:
            train_dense_model(spark, input_col, model, parallelism)
