"""
Stacking: combinación de las familias de features con un meta-modelo.

    1. Modelos base: el mejor modelo de cada familia según su AUC-PR en val (nunca en test).
    2. Cada texto de val y test queda descrito por el score de cada modelo base.
    3. Meta-modelo: StandardScaler + LogisticRegression sobre esos scores, entrenado en val.
       En val los scores son "honestos" (los modelos base no vieron esos textos); en train
       serían optimistas y el meta-modelo confiaría de más en los que más sobreajustan.
    4. Se evalúa en test con las mismas métricas que los modelos base (evaluate.py).

Además:
    Pesos:     coeficientes del meta-modelo sobre scores estandarizados (comparables entre sí)
    Ablación:  el meta-modelo se reentrena quitando cada familia; la caída de AUC-PR en test
               mide lo que aporta esa familia que las demás no tienen

Salida: fila "stacking" en results/metrics.csv y results/by_subsource.csv, y
        results/stacking_weights.csv, results/stacking_ablation.csv
"""

import json
import time

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession

from config import RESULTS_DIR
from evaluate import (best_f1_threshold, by_subsource, metrics_row, predict, to_score,
                      trained_models)
from train import add_stratified_folds

# Familia de cada columna de features (una columna nueva cuenta como su propia familia)
FAMILIES = {"tf": "lexical", "tf_nostop": "lexical", "tf_char": "char",
            "features_style": "style", "features_lm": "lm"}

EVALUATOR = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score",
                                          metricName="areaUnderPR")



def select_base_models(spark: SparkSession) -> dict:
    """
    Elige el mejor modelo de cada familia por su AUC-PR en val.

    Args:
        spark: SparkSession

    Returns:
        dict
            {familia: nombre del modelo}
    """
    best = {}
    for name in trained_models():
        family = FAMILIES.get(name.rsplit("_", 1)[0], name.rsplit("_", 1)[0])
        auc_pr = EVALUATOR.evaluate(predict(spark, name, "val"))
        print(f"[stacking] {name} ({family}): AUC-PR val = {auc_pr:.4f}")
        if family not in best or auc_pr > best[family][1]:
            best[family] = (name, auc_pr)

    if len(best) < 2:
        raise ValueError("El stacking necesita modelos de al menos dos familias")
    return {family: name for family, (name, _) in best.items()}



def score_table(spark: SparkSession, names: list, split: str) -> DataFrame:
    """
    Una fila por texto con el score de cada modelo base.

    Args:
        spark: SparkSession
        names: list
            Modelos base
        split: str
            "val" o "test"

    Returns:
        DataFrame
            (id, label, <una columna por modelo base con su score>)
    """
    table = None
    for name in names:
        scores = predict(spark, name, split).withColumnRenamed("score", name)
        table = scores if table is None else table.join(scores.drop("label"), on="id", how="inner")
    return table



def fit_meta(val: DataFrame, cols: list, num_folds: int = 3) -> PipelineModel:
    """
    Entrena el meta-modelo sobre los scores de val, eligiendo regParam con un
    CrossValidator estratificado (los mismos folds que en train.py).

    El escalado es necesario: cada modelo base tiene su propia escala de score
    (log-odds en NB y LR, margen en SVC...).

    Args:
        val: DataFrame
            Salida de score_table(..., "val")
        cols: list
            Columnas de score usadas (modelos base)
        num_folds: int
            Número de folds de la validación cruzada

    Returns:
        PipelineModel
            VectorAssembler -> StandardScaler -> LogisticRegression, ajustado con todo val
    """
    lr = LogisticRegression(featuresCol="scaled", labelCol="label")
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=cols, outputCol="base_scores"),
        StandardScaler(inputCol="base_scores", outputCol="scaled", withMean=True, withStd=True),
        lr,
    ])
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01, 0.1]).build()
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                              metricName="areaUnderPR")

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator,
                        numFolds=num_folds, foldCol="fold")
    # bestModel ya está reentrenado con todo val
    return cv.fit(add_stratified_folds(val, num_folds)).bestModel



def ablation(val: DataFrame, test: DataFrame, base: dict, full_auc_pr: float) -> pd.DataFrame:
    """
    Reentrena el meta-modelo quitando cada familia y mide la caída de AUC-PR en test.

    Args:
        val: DataFrame
            Salida de score_table(..., "val")
        test: DataFrame
            Salida de score_table(..., "test")
        base: dict
            {familia: modelo base}, salida de select_base_models
        full_auc_pr: float
            AUC-PR en test del stacking con todas las familias

    Returns:
        DataFrame de pandas
            (removed_family, auc_pr_test, drop): drop = full_auc_pr - auc_pr_test
    """
    rows = []
    for family in base:
        cols = [name for fam, name in base.items() if fam != family]
        meta = fit_meta(val, cols)
        auc_pr = EVALUATOR.evaluate(to_score(meta.transform(test)))
        rows.append({"removed_family": family, "auc_pr_test": auc_pr, "drop": full_auc_pr - auc_pr})
        print(f"[stacking] sin {family}: AUC-PR test = {auc_pr:.4f} (caída {full_auc_pr - auc_pr:+.4f})")
    return pd.DataFrame(rows).sort_values("drop", ascending=False)



def append_results(file: str, new: pd.DataFrame, sort_by: str = None) -> None:
    """
    Añade las filas del stacking a un CSV de evaluate.py, sustituyendo las de una
    ejecución anterior.

    Args:
        file: str
            "metrics.csv" o "by_subsource.csv"
        new: DataFrame de pandas
            Filas nuevas (columna model = "stacking")
        sort_by: str
            Columna por la que ordenar (descendente), o None
    """
    path = RESULTS_DIR / file
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["model"] != "stacking"], new], ignore_index=True)
    if sort_by:
        new = new.sort_values(sort_by, ascending=False)
    new.to_csv(path, index=False)



def run_stacking(spark: SparkSession) -> pd.DataFrame:
    """
    Stacking completo: selección de modelos base, meta-modelo, evaluación, pesos y ablación.

    Args:
        spark: SparkSession

    Returns:
        DataFrame de pandas
            La fila del stacking en metrics.csv
    """
    base = select_base_models(spark)
    cols = list(base.values())
    print(f"[stacking] modelos base: {base}")

    # Se cachean: el meta-modelo, el umbral, las métricas y la ablación las recorren varias veces
    val = score_table(spark, cols, "val").cache()
    test = score_table(spark, cols, "test").cache()

    start = time.perf_counter()
    meta = fit_meta(val, cols)
    tune_s = time.perf_counter() - start

    val_scores = to_score(meta.transform(val)).cache()
    test_scores = to_score(meta.transform(test)).cache()

    # El umbral se elige en val, que también entrenó al meta-modelo. El sesgo es mínimo: el
    # meta-modelo solo tiene un peso por familia frente a cientos de miles de textos de val
    threshold = best_f1_threshold(val_scores)
    lr_model = meta.stages[-1]
    weights = dict(zip(cols, lr_model.coefficients.toArray().tolist()))

    row = {"model": "stacking", "features": "+".join(base), "classifier": "lr"}
    row.update(metrics_row(val_scores, test_scores, threshold))
    row.update({"tune_s": tune_s, "params": json.dumps({"regParam": lr_model.getRegParam(),
                                                         "weights": weights})})
    print(f"[stacking] AUC-PR test = {row['auc_pr_test']:.4f} "
          f"(línea base {row['prevalence_test']:.4f}) | F1 test = {row['f1_test']:.4f}")

    sources = by_subsource(spark, test_scores, threshold).toPandas()
    sources.insert(0, "model", "stacking")

    # Pesos sobre scores estandarizados: comparables entre familias
    family_of = {name: fam for fam, name in base.items()}
    weights_df = pd.DataFrame([{"family": family_of[name], "model": name, "weight": w}
                               for name, w in weights.items()]).sort_values("weight", ascending=False)

    ablation_df = ablation(val, test, base, row["auc_pr_test"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    append_results("metrics.csv", pd.DataFrame([row]), sort_by="auc_pr_test")
    append_results("by_subsource.csv", sources)
    weights_df.to_csv(RESULTS_DIR / "stacking_weights.csv", index=False)
    ablation_df.to_csv(RESULTS_DIR / "stacking_ablation.csv", index=False)

    for df in (val, test, val_scores, test_scores):
        df.unpersist()
    return pd.DataFrame([row])
