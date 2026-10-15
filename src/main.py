"""
Procesamiento de datos...
"""
import os

from clean_data import build_clean
from config import CLEAN_PATH, LEXICAL_PATH, get_spark
from features_lexical import LexicalFamily



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

    # 2) Loading the clean data and start building the n-grams of the texts.
    df_clean = spark.read.parquet(str(CLEAN_PATH))

    if not data_already_exists(LEXICAL_PATH):
        LexicalFamily().save(df_clean)


if __name__ == "__main__":
    main()
