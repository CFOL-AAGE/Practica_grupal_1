"""
Implementation of the first family: Lexical

The IDF is not pre-computed in the training data, but it's implemented when
using the cross-validation in order to avoid the phenomena of data leakage!
"""

from pyspark.ml.feature import HashingTF, NGram, RegexTokenizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F



class LexicalFamily:
    """Extractor of lexical features

    It extracts the 1-grams and 2-grams of the texts that survived the cleaning
    process executed by the script `clean_data`, in order to create a sparse
    vector whose indexes are defined by a hash function (HashingTF) and their
    values correspond to the term frequency (tf). As all the texts have the same
    length, the absolute tf is stored.

    Attributes:
        num_features: int
            Sparse vector size

    Methods:
        _concat_terms(self, df: Dataframe) -> DataFrame
            Concat what we have obtained from the tokenizer and the Ngram(n=2)
        transform(self, df: DataFrame) -> DataFrame
            Returns the lexical features table: (id, tf)

    """


    def __init__(self, num_features: int = 2 ** 18):
        self.num_features = num_features

        # Instance a tokenizer to create the 1-grams. The punctuation is dropped.
        self.tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens",
                                        pattern=r"[\w']+", gaps=False, toLowercase=True)

        # 2-grams from consecutive tokens, joined by a space: "w1 w2"
        self.ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")

        # Each term goes to the slot murmur3(term) mod num_features
        # binary=false --> value = raw count
        self.hashing_tf = HashingTF(inputCol="terms", outputCol="tf",
                                    numFeatures=num_features, binary=False)


    def _concat_terms(self, df: DataFrame) -> DataFrame:
        """
        Adds the column `terms`: the 1-grams followed by the 2-grams of each text.

        Args:
            df: Dataframe
                Clean Dataframe

        Returns:
            DataFrame
                With a new column added for terms.
        """
        # Genrate 1-grams and 2-grams
        df = self.tokenizer.transform(df)
        df = self.ngram.transform(df)

        # Add new column for terms
        df_terms = df.withColumn("terms", F.concat("tokens", "bigrams"))
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
                Columns `id` and `tf` (SparseVector of size num_features)
        """
        df_tf = self.hashing_tf.transform(self._concat_terms(df))
        return df_tf.select("id", "tf")
