from pyspark.ml.feature import (HashingTF, NGram,
                                RegexTokenizer, StopWordsRemover)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import LEXICAL_PATH


class LexicalFamily:
    """Extractor of lexical features

    It extracts the 1-grams and 2-grams of the texts that survived the cleaning
    process executed by the script `clean_data`, in order to create a sparse
    vector whose indexes are defined by a hash function (HashingTF) and their
    values correspond to the term frequency (tf). As all the texts have the same
    length, the absolute tf is stored.

    Two variants are built from the same tokens, so they can be compared:
        tf:        1-grams + 2-grams keeping the stop words
        tf_nostop: 1-grams + 2-grams after removing the stop words (StopWordsRemover)
    Both use the same HashingTF, so a term falls in the same slot in both vectors.

    Output table (LEXICAL_PATH), one row per text of train, val and test:
        id (long), tf (SparseVector), tf_nostop (SparseVector)

    Attributes:
        num_features: int
            Sparse vector size
        hashing_tf: HashingTF
            Shared hashing transformer (terms -> tf, raw counts)

    Methods:
        _apply_configs(self, df: DataFrame) -> DataFrame
            Applies the tokenizer, the stop words remover and the NGram(n=2) of
            both variants
        _concat_terms(self, df: DataFrame) -> DataFrame
            Concats the 1-grams and the 2-grams of each variant
        transform(self, df: DataFrame) -> DataFrame
            Returns the lexical features table: (id, tf, tf_nostop)
        save(self, df: DataFrame) -> None
            Writes the lexical features table to LEXICAL_PATH, excluding lm_fit
    """


    def __init__(self, num_features: int = 2 ** 18):
        self.num_features = num_features
        # binary=false --> value = raw count
        self.hashing_tf = HashingTF(inputCol="terms", outputCol="tf",
                                    numFeatures=num_features, binary=False)


    def _apply_configs(self, df: DataFrame) -> DataFrame:
        """
        Apply the two approaches in order to get the terms of the texts using
        StopWords and not.

        Args:
            df: DataFrame
                Clean DataFrame (needs the column `text`)

        Returns:
            DataFrame
                Clean DataFrame with the array columns `tokens`, `bigrams`,
                `tokens_nostop` and `bigrams_nostop` added.
        """
        # WITHOUT SW:
        # Instance a tokenizer to create the 1-grams. The punctuation is dropped.
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens",
                                        pattern=r"[\w']+", gaps=False, toLowercase=True)
        # 2-grams from consecutive tokens, joined by a space: "w1 w2"
        ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")

        # WITH SW:
        stopwords = StopWordsRemover(inputCol="tokens", outputCol="tokens_nostop")
        ngram_nostop = NGram(n=2, inputCol="tokens_nostop", outputCol="bigrams_nostop")

        # Iterate over all the transfomers (don't learn anything): the order matters!
        transformers = [tokenizer, ngram, stopwords, ngram_nostop]
        for tr in transformers:
            df = tr.transform(df)      # each one works on the output of the previous one

        return df


    def _concat_terms(self, df: DataFrame) -> DataFrame:
        """
        Adds the columns `terms` and `terms_nostop`: the 1-grams followed by the
        2-grams of each text, with and without stop words.

        Args:
            df: DataFrame
                Clean DataFrame

        Returns:
            DataFrame
                With the array columns `terms` and `terms_nostop` added.
        """
        # Add new columns for terms
        df_terms = (
            self._apply_configs(df).
                withColumn("terms", F.concat("tokens", "bigrams")).
                withColumn("terms_nostop", F.concat("tokens_nostop", "bigrams_nostop"))
        )

        return df_terms


    def transform(self, df: DataFrame) -> DataFrame:
        """
        Builds the lexical features (multi-dimensional sparse vector)
        of each text.

        Args:
            df: DataFrame
                Clean DataFrame

        Returns:
            DataFrame
                Columns `id`, `tf` and `tf_nostop` (SparseVectors of size num_features)
        """
        df_tf = self.hashing_tf.transform(self._concat_terms(df))

        # Same HashingTF (same slots for the same term), only the columns change
        df_tf = self.hashing_tf.transform(df_tf, {self.hashing_tf.inputCol: "terms_nostop",
                                                  self.hashing_tf.outputCol: "tf_nostop"})
        return df_tf.select("id", "tf", "tf_nostop")


    def save(self, df: DataFrame) -> None:
        """
        Writes the lexical features of every split except lm_fit to LEXICAL_PATH.

        Args:
            df: DataFrame
                Clean DataFrame (all the splits)
        """
        # lm_fit is reserved for the n-gram language model (never used by the classifiers)
        df = df.filter(F.col("split") != "lm_fit")
        (
            self.transform(df).
                write.mode("overwrite").
                parquet(str(LEXICAL_PATH))
        )
