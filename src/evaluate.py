"""
Evaluación sobre val y test de los modelos guardados por train.py (models/<input_col>_<modelo>).

La clase positiva es 1.0 (IA). Métricas:
    Sin umbral:          AUC-PR (principal, junto a su línea base = prevalencia de IA),
                         AUC-ROC y TPR con FPR = 1 %
    Punto de operación:  umbral que maximiza el F1 en val, aplicado tal cual en test:
                         precisión, recall, F1, FPR sobre humanos, accuracy y matriz de confusión
    Validez:             recall de IA por generador y FPR por fuente humana (subsource), en test
    Coste:               tiempos de ajuste y de entrenamiento (guardados por train.py en train_info.json)

Cada modelo se evalúa sobre su `score` = rawPrediction[1] - rawPrediction[0]: existe en los
cinco clasificadores (LinearSVC no da probabilidades) y crece con la confianza en IA.

Salida: results/metrics.csv (una fila por modelo) y results/by_subsource.csv
"""

import json

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Bucketizer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config import CLEAN_PATH, MODELS_DIR, RESULTS_DIR
from load_datasets import load_split
from train import DENSE_PATHS, FEATURE_PATHS

# Parquet donde vive cada columna de features (familias 1 a 4)
ALL_PATHS = {**FEATURE_PATHS, **DENSE_PATHS}

N_CANDIDATES = 1000     # umbrales candidatos de la búsqueda en val (cuantiles del score)
MAX_FPR = 0.01          # FPR de la métrica "TPR con FPR = 1 %"
REL_ERROR = 1e-4        # error relativo de approxQuantile



def trained_models() -> list:
    """
    Modelos entrenados por completo (con _SUCCESS) en models/.

    Returns:
        list
            Nombres de las carpetas: "<input_col>_<modelo>" (p. ej. "tf_nostop_lr")
    """
    names = sorted(p.name for p in MODELS_DIR.iterdir() if (p / "_SUCCESS").exists())
    if not names:
        raise FileNotFoundError(f"No hay modelos entrenados en {MODELS_DIR}: ejecutar train.py antes")
    return names



def to_score(predictions: DataFrame) -> DataFrame:
    """
    Reduce la salida de un modelo a su score.

    Args:
        predictions: DataFrame
            Salida de PipelineModel.transform (necesita id, label y rawPrediction)

    Returns:
        DataFrame
            (id, label, score), con score = rawPrediction[1] - rawPrediction[0]
    """
    raw = vector_to_array(F.col("rawPrediction"))
    return predictions.select("id", "label", (raw[1] - raw[0]).alias("score"))



def predict(spark: SparkSession, name: str, split: str) -> DataFrame:
    """
    Aplica un modelo guardado a un split.

    Args:
        spark: SparkSession
        name: str
            Carpeta del modelo dentro de models/: "<input_col>_<modelo>"
        split: str
            "val" o "test"

    Returns:
        DataFrame
            (id, label, score)
    """
    input_col = name.rsplit("_", 1)[0]
    model = PipelineModel.load(str(MODELS_DIR / name))
    df = load_split(spark, ALL_PATHS[input_col], split).select("id", input_col, "label")
    return to_score(model.transform(df))



def threshold_free(df: DataFrame) -> dict:
    """
    Métricas que no dependen del umbral.

    Args:
        df: DataFrame
            (id, label, score)

    Returns:
        dict
            n, prevalence (línea base de la AUC-PR), auc_pr, auc_roc, tpr_at_fpr1
    """
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score")
    counts = df.agg(F.count("*").alias("n"), F.mean("label").alias("prevalence")).first()

    # TPR con FPR = 1 %: umbral = percentil 99 de los scores de los humanos
    t = df.filter(F.col("label") == 0.0).approxQuantile("score", [1 - MAX_FPR], REL_ERROR)[0]
    tpr = df.filter(F.col("label") == 1.0).agg(F.mean((F.col("score") > t).cast("double"))).first()[0]

    return {
        "n": counts["n"],
        "prevalence": counts["prevalence"],
        "auc_pr": evaluator.evaluate(df, {evaluator.metricName: "areaUnderPR"}),
        "auc_roc": evaluator.evaluate(df, {evaluator.metricName: "areaUnderROC"}),
        "tpr_at_fpr1": tpr,
    }



def best_f1_threshold(val: DataFrame) -> float:
    """
    Umbral del score que maximiza el F1 de la clase IA en val.

    Los candidatos son N_CANDIDATES cuantiles del score: un Bucketizer + groupBy da los
    conteos de cada intervalo, y el driver solo acumula N_CANDIDATES filas (sin ordenar todo).

    Args:
        val: DataFrame
            (id, label, score) de val

    Returns:
        float
            score >= umbral -> IA
    """
    probs = [i / N_CANDIDATES for i in range(1, N_CANDIDATES)]
    candidates = sorted(set(val.approxQuantile("score", probs, REL_ERROR)))

    # El intervalo j >= 1 contiene los scores de [candidates[j - 1], candidates[j])
    splits = [float("-inf")] + candidates + [float("inf")]
    buckets = (
        Bucketizer(splits=splits, inputCol="score", outputCol="bucket", handleInvalid="keep").
            transform(val).
            groupBy("bucket").
            agg(F.sum("label").alias("pos"), F.count("*").alias("n")).
            collect()
    )
    pos = [0] * (len(splits) - 1)
    n = [0] * (len(splits) - 1)
    for row in buckets:
        if int(row["bucket"]) < len(pos):           # los scores NaN (intervalo extra) se descartan
            pos[int(row["bucket"])] = int(row["pos"])
            n[int(row["bucket"])] = row["n"]
    total_pos = sum(pos)

    # Del umbral más alto al más bajo: con candidates[k], los intervalos k + 1 .. fin son IA
    best_f1, best_t, tp, fp = -1.0, candidates[-1], 0, 0
    for k in range(len(candidates) - 1, -1, -1):
        tp += pos[k + 1]
        fp += n[k + 1] - pos[k + 1]
        f1 = 2 * tp / (2 * tp + fp + (total_pos - tp)) if tp else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, candidates[k]
    return best_t



def threshold_metrics(df: DataFrame, threshold: float) -> dict:
    """
    Métricas del clasificador con un umbral fijo.

    Args:
        df: DataFrame
            (id, label, score)
        threshold: float
            Salida de best_f1_threshold (elegido en val)

    Returns:
        dict
            tp, fp, tn, fn, precision, recall, f1, fpr, accuracy
    """
    ai = F.col("score") >= threshold
    is_ai = F.col("label") == 1.0
    count = lambda cond: F.sum(F.when(cond, 1).otherwise(0))

    c = df.agg(count(ai & is_ai).alias("tp"), count(ai & ~is_ai).alias("fp"),
               count(~ai & ~is_ai).alias("tn"), count(~ai & is_ai).alias("fn")).first()
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

    div = lambda a, b: a / b if b else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "fpr": div(fp, fp + tn),
        "accuracy": div(tp + tn, tp + fp + tn + fn),
    }



def metrics_row(val: DataFrame, test: DataFrame, threshold: float) -> dict:
    """
    Todas las métricas de un modelo a partir de sus scores en val y test.

    Args:
        val: DataFrame
            (id, label, score) de val
        test: DataFrame
            (id, label, score) de test
        threshold: float
            Salida de best_f1_threshold(val)

    Returns:
        dict
            Métricas sin umbral de val y test (sufijos _val, _test) y con umbral de test
    """
    row = {"threshold": threshold}
    row.update({f"{k}_val": v for k, v in threshold_free(val).items()})
    row.update({f"{k}_test": v for k, v in threshold_free(test).items()})
    row.update({f"{k}_test": v for k, v in threshold_metrics(test, threshold).items()})
    return row



def by_subsource(spark: SparkSession, test: DataFrame, threshold: float) -> DataFrame:
    """
    Proporción de textos clasificados como IA por subsource de test: recall en los
    generadores de IA y FPR en las fuentes humanas.

    Args:
        spark: SparkSession
        test: DataFrame
            (id, label, score) de test
        threshold: float
            Salida de best_f1_threshold (elegido en val)

    Returns:
        DataFrame
            (subsource, label, n, rate, metric) con metric = "recall" (IA) o "fpr" (humano)
    """
    sources = (
        spark.read.parquet(str(CLEAN_PATH)).
            filter(F.col("split") == "test").
            select("id", "subsource")
    )
    return (
        test.join(F.broadcast(sources), on="id", how="inner").
            groupBy("subsource", "label").
            agg(F.count("*").alias("n"),
                F.mean((F.col("score") >= threshold).cast("double")).alias("rate")).
            withColumn("metric", F.when(F.col("label") == 1.0, "recall").otherwise("fpr")).
            orderBy("label", "subsource")
    )



def evaluate_model(spark: SparkSession, name: str) -> tuple:
    """
    Todas las métricas de un modelo guardado.

    Args:
        spark: SparkSession
        name: str
            Carpeta del modelo dentro de models/

    Returns:
        tuple
            (dict con su fila de metrics.csv, DataFrame de pandas con sus filas de by_subsource.csv)
    """
    input_col, model = name.rsplit("_", 1)
    val = predict(spark, name, "val").cache()
    test = predict(spark, name, "test").cache()

    threshold = best_f1_threshold(val)
    row = {"model": name, "features": input_col, "classifier": model}
    row.update(metrics_row(val, test, threshold))

    # Escrito por train.py: AUC-PR de la CV, hiperparámetros y tiempos
    info_path = MODELS_DIR / name / "train_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        row.update({"auc_pr_cv": info["auc_pr_cv"], "tune_s": info["tune_s"],
                    "fit_s": info["fit_s"], "params": json.dumps(info["params"])})

    sources = by_subsource(spark, test, threshold).toPandas()
    sources.insert(0, "model", name)

    val.unpersist()
    test.unpersist()
    return row, sources



def evaluate_all(spark: SparkSession) -> pd.DataFrame:
    """
    Evalúa todos los modelos guardados en models/ y escribe results/metrics.csv y
    results/by_subsource.csv.

    Args:
        spark: SparkSession

    Returns:
        DataFrame de pandas
            metrics.csv, ordenado por AUC-PR en test
    """
    rows, sources = [], []
    for name in trained_models():
        row, src = evaluate_model(spark, name)
        print(f"[{name}] AUC-PR test = {row['auc_pr_test']:.4f} "
              f"(línea base {row['prevalence_test']:.4f}) | F1 test = {row['f1_test']:.4f}")
        rows.append(row)
        sources.append(src)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows).sort_values("auc_pr_test", ascending=False)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pd.concat(sources).to_csv(RESULTS_DIR / "by_subsource.csv", index=False)
    return metrics
