"""
Loading of the datasets used by the models: features of one family + label, per split.

Used by the sparse models (families 1 and 2: lexical.parquet and char.parquet).
Valid for any family that follows the team contract: a parquet with `id` and
its feature columns, joined with clean.parquet by `id`.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config import CLEAN_PATH



def load_split(spark: SparkSession, features_path, split: str) -> DataFrame:
    """
    Joins the features of one family with their labels (human, AI)

    Args:
        spark: SparkSession
        features_path: Path
            Parquet of the family (LEXICAL_PATH, CHAR_PATH...)
        split: str
            "train", "val" or "test"

    Returns:
        DataFrame
            Feature columns of the family + `label`
    """
    # Small side: no text, no vectors. Partition pruning: only the folder split=<split> is read
    labels = (
        spark.read.parquet(str(CLEAN_PATH)).
            filter(F.col("split") == split).
            select("id", "label")
    )
    features = spark.read.parquet(str(features_path))

    # Broadcast join: the small side is copied to every executor, so the big
    # side (feature vectors) is joined where it is, without a shuffle
    return features.join(F.broadcast(labels), on="id", how="inner")
