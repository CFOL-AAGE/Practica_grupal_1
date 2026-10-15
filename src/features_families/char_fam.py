from pyspark.ml.feature import (HashingTF, NGram,
                                RegexTokenizer, StopWordsRemover)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import CHAR_PATH


class CharFamily:
    """Extractor of character features

    It extracts the 3-grams, 4-grams and 5-grams of characters of the texts that
    survived the cleaning process, keeping spaces, punctuation and capital letters
    (they carry the form of the writing). As in LexicalFamily, the raw counts are
    stored in a sparse vector whose indexes are defined by HashingTF.

    MLlib has no character n-grams: the RegexTokenizer splits the text into single
    characters and NGram joins n consecutive ones with a space ("the" -> "t h e").

    Output table (CHAR_PATH), one row per text of train, val and test:
        id (long), tf_char (SparseVector)

    Attributes:
        num_features: int
            Sparse vector size
        ngram_sizes: tuple
            Sizes of the character n-grams
        hashing_tf: HashingTF
            Hashing transformer (char_terms -> tf_char, raw counts)

    Methods:
        _apply_configs(self, df: DataFrame) -> DataFrame
            Applies the character tokenizer and the NGram(n) of every size
        _concat_terms(self, df: DataFrame) -> DataFrame
            Concats the n-grams of every size
        transform(self, df: DataFrame) -> DataFrame
            Returns the character features table: (id, tf_char)
        save(self, df: DataFrame) -> None
            Writes the character features table to CHAR_PATH, excluding lm_fit
    """


    def __init__(self, num_features: int = 2 ** 18, ngram_sizes: tuple = (3, 4, 5)):
        self.num_features = num_features
        self.ngram_sizes = ngram_sizes
        # binary=false --> value = raw count
        self.hashing_tf = HashingTF(inputCol="char_terms", outputCol="tf_char",
                                    numFeatures=num_features, binary=False)


    def _apply_configs(self, df: DataFrame) -> DataFrame:
        """
        Splits the text into characters and builds their n-grams of every size.

        Args:
            df: DataFrame
                Clean DataFrame (needs the column `text`)

        Returns:
            DataFrame
                Clean DataFrame with the array columns `chars` and `chars_<n>`
                (one per size in ngram_sizes) added.
        """
        # Every character is a token (spaces and punctuation included), case is kept
        tokenizer = RegexTokenizer(inputCol="text", outputCol="chars",
                                   pattern=".", gaps=False, toLowercase=False)
        # One NGram per size: chars_3, chars_4, chars_5
        ngrams = [NGram(n=n, inputCol="chars", outputCol=f"chars_{n}")
                  for n in self.ngram_sizes]

        # Iterate over all the transformers (don't learn anything): the order matters!
        for tr in [tokenizer, *ngrams]:
            df = tr.transform(df)      # each one works on the output of the previous one

        return df


    def _concat_terms(self, df: DataFrame) -> DataFrame:
        """
        Adds the column `char_terms`: the character n-grams of every size.

        Args:
            df: DataFrame
                Clean DataFrame

        Returns:
            DataFrame
                With the array column `char_terms` added.
        """
        ngram_cols = [f"chars_{n}" for n in self.ngram_sizes]
        return self._apply_configs(df).withColumn("char_terms", F.concat(*ngram_cols))


    def transform(self, df: DataFrame) -> DataFrame:
        """
        Builds the character features (multi-dimensional sparse vector)
        of each text.

        Args:
            df: DataFrame
                Clean DataFrame

        Returns:
            DataFrame
                Columns `id` and `tf_char` (SparseVector of size num_features)
        """
        df_tf = self.hashing_tf.transform(self._concat_terms(df))
        return df_tf.select("id", "tf_char")


    def save(self, df: DataFrame) -> None:
        """
        Writes the character features of every split except lm_fit to CHAR_PATH.

        Args:
            df: DataFrame
                Clean DataFrame (all the splits)
        """
        # lm_fit is reserved for the n-gram language model (never used by the classifiers)
        df = df.filter(F.col("split") != "lm_fit")
        (
            self.transform(df).
                write.mode("overwrite").
                parquet(str(CHAR_PATH))
        )
