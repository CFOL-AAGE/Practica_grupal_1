"""
Procesamiento de datos...
"""
import os

from clean_data import build_clean
from config import (CHAR_PATH, CLEAN_PATH, LEXICAL_PATH,
                    LM_PATH, STYLE_PATH, get_spark)
from evaluate import evaluate_all
from features_families import (CharFamily, LexicalFamily, build_features,
                               build_style_features, build_table, count_ngrams)
from stacking import run_stacking
from train import train_all_dense, train_all_sparse



def data_already_exists(path) -> bool:
    """
    Verifica si el parquet ya existe y se escribió completo.
    Spark crea _SUCCESS solo al terminar la escritura sin errores.
    """
    return os.path.exists(path / "_SUCCESS")

def main():
    # 1) Loading the cleaning data pipeline and executing it.
    spark = get_spark()

    if not data_already_exists(CLEAN_PATH): # Save 5 minutes of time execution.
        build_clean(spark)

    # Loading the clean data
    df_clean = spark.read.parquet(str(CLEAN_PATH))

    # 2) Lexical Family
    if not data_already_exists(LEXICAL_PATH):
        LexicalFamily().save(df_clean)

    # 3) Char Family
    if not data_already_exists(CHAR_PATH):
        CharFamily().save(df_clean)

    # 4) LM family
    if not data_already_exists(LM_PATH):
        count_ngrams(spark)
        build_table(spark)
        build_features(spark)

    # 5) Style Family
    if not data_already_exists(STYLE_PATH):
        build_style_features(spark)

    # 6) Modelos sobre las features dispersas (familias 1, 2) y densas (familias 3, 4)
    # (los ya guardados se saltan)
    train_all_sparse(spark)
    train_all_dense(spark)

    # 7) Evaluación en val y test -> results/metrics.csv, results/by_subsource.csv
    evaluate_all(spark)

    # 8) Stacking del mejor modelo de cada familia -> fila "stacking" en results/metrics.csv,
    # results/stacking_weights.csv y results/stacking_ablation.csv
    run_stacking(spark)


if __name__ == "__main__":
    main()
