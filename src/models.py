"""
Definitions of the MLlib models of the project: not fitted Pipelines and their
grids of hyperparameters. The training lives in train.py.

Sparse features (families 1 and 2: lexical.parquet and char.parquet):
    build_sparse_pipeline, build_sparse_grid
    Linear models over the HashingTF raw counts: NaiveBayes, LogisticRegression
    and LinearSVC (the last two with the IDF inside the Pipeline).

Dense features (families 3 and 4: style.parquet and lm.parquet):
    build_dense_pipeline, build_dense_grid
    LogisticRegression (StandardScaler inside the Pipeline) as linear baseline,
    and GBTClassifier for non-linear interactions (no scaling needed).
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression, NaiveBayes, GBTClassifier
from pyspark.ml.feature import IDF, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder

from config import SEED


def build_sparse_pipeline(input_col: str, model: str) -> Pipeline:
    """
    Builds the (not fitted) Pipeline of one model over one sparse column of the
    families 1 and 2.

    Args:
        input_col: str
            Raw counts column: "tf", "tf_nostop" or "tf_char"
        model: str
            "nb" (NaiveBayes), "lr" (LogisticRegression) or "svc" (LinearSVC)

    Returns:
        Pipeline
            nb:      NaiveBayes over the raw counts
            lr, svc: IDF (Estimator, fitted inside each fit) -> classifier over TF-IDF
    """
    classifiers = {
        "lr": LogisticRegression(featuresCol="features", labelCol="label"),
        "svc": LinearSVC(featuresCol="features", labelCol="label"),
        "nb": NaiveBayes(featuresCol=input_col, labelCol="label", modelType="multinomial")
    }
    pip_stages = []

    model_selected = classifiers[model]

    # Multinomial Naive Bayes models counts: it works directly on the raw tf
    if model == "nb":
        pip_stages.append(model_selected)
    else:
        idf = IDF(inputCol=input_col, outputCol="features") # IDF(t, d, D)
        pip_stages.extend([idf, model_selected])

    return Pipeline(stages=pip_stages)



def build_sparse_grid(pipeline: Pipeline, model: str) -> list:
    """
    Builds the grid of hyperparameters of the classifier of a sparse Pipeline
    (families 1 and 2).

    Each combination costs numFolds fits, so the grids are kept small.

    Args:
        pipeline: Pipeline
            Output of build_sparse_pipeline
        model: str
            "nb", "lr" or "svc"

    Returns:
        list
            One ParamMap (dict Param -> value) per combination
    """
    clf = pipeline.getStages()[-1]      # the classifier is always the last stage
    grid = ParamGridBuilder()

    if model == "nb":
        grid.addGrid(clf.smoothing, [0.1, 0.5, 1.0])                    # 3 combinations
    elif model == "lr":
        grid.addGrid(clf.regParam, [1e-4, 1e-3, 1e-2])                  # 3 x 2 = 6
        grid.addGrid(clf.elasticNetParam, [0.0, 0.5])                   # 0 = L2, 0.5 = L1 + L2
    elif model == "svc":
        grid.addGrid(clf.regParam, [1e-4, 1e-3, 1e-2])                  # 3

    return grid.build()


def build_dense_pipeline(input_col: str, model: str) -> Pipeline:
    """
    Builds the (not fitted) Pipeline of one model over one dense column of the
    families 3 and 4.

    Args:
        input_col: str
            Dense features column (assembled vector)
        model: str
            "lr" (LogisticRegression) or "gbt" (GBTClassifier)

    Returns:
        Pipeline
            lr:  StandardScaler (Estimator, fitted inside each fit) -> LogisticRegression
            gbt: GBTClassifier directly over the features (trees do not need scaling)
    """
    classifiers = {
        "lr": LogisticRegression(featuresCol="scaled", labelCol="label", maxIter=100),
        "gbt": GBTClassifier(featuresCol=input_col, labelCol="label", seed=SEED)
    }
    pip_stages = []

    model_selected = classifiers[model]

    # The scaler learns mean and std: inside the Pipeline it only sees the train of each fold
    if model == "lr":
        scaler = StandardScaler(inputCol=input_col, outputCol="scaled", withMean=True, withStd=True)
        pip_stages.extend([scaler, model_selected])
    else:
        pip_stages.append(model_selected)

    return Pipeline(stages=pip_stages)



def build_dense_grid(pipeline: Pipeline, model: str) -> list:
    """
    Builds the grid of hyperparameters of the classifier of a dense Pipeline
    (families 3 and 4).

    Args:
        pipeline: Pipeline
            Output of build_dense_pipeline
        model: str
            "lr" or "gbt"

    Returns:
        list
            One ParamMap (dict Param -> value) per combination
    """
    clf = pipeline.getStages()[-1]      # the classifier is always the last stage
    grid = ParamGridBuilder()

    if model == "lr":
        grid.addGrid(clf.regParam, [0.0, 0.01, 0.1])                   # regularización
        grid.addGrid(clf.elasticNetParam, [0.0, 1.0])                  # 0 = L2, 1 = L1
    elif model == "gbt":
        grid.addGrid(clf.maxDepth, [3, 5])                             # 2 x 2 = 4
        grid.addGrid(clf.maxIter, [50, 100])                           # number of trees

    return grid.build()
