"""
Procesamiento de datos...
"""
import os

from clean_data import build_clean
from config import CHAR_PATH, CLEAN_PATH, LEXICAL_PATH, get_spark
from features_families import CharFamily, LexicalFamily
from train import train_all_sparse



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

    # 4) Models over the sparse features (the ones already saved are skipped)
    train_all_sparse(spark)


if __name__ == "__main__":
    main()
