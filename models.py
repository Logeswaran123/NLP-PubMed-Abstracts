"""
Models
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

MAX_VOCAB_LENGTH = 10000
MAX_INPUT_LENGTH = 15


def Create_Text_Vectorizer(train_sentences):
    text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_LENGTH,
                            output_mode="int",
                            output_sequence_length=MAX_INPUT_LENGTH)
    text_vectorizer.adapt(train_sentences)
    return text_vectorizer


def Create_Embedding_Layer():
    embedding = layers.Embedding(input_dim=MAX_VOCAB_LENGTH, # set input shape
                            output_dim=128, # set size of embedding vector
                            embeddings_initializer="uniform", # default, intialize randomly
                            input_length=MAX_INPUT_LENGTH, # how long is each input
                            name="embedding_1")
    return embedding


class Model():
    """
    Class with multiple model architectures
    """
    def __init__(self) -> None:
        pass

    def Baseline(self):
        """
        Create and return a Baseline model
        """
        pass

    def Model_1(self, train_sentences):
        """
        Create and return a simple Dense Model
        """
        pass