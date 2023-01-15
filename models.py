"""
Models
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

MAX_VOCAB_LENGTH = 68000
MAX_OUTPUT_LENGTH = 55


def Create_Text_Vectorizer(train_sentences):
    text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_LENGTH,  # number of words in vocabulary
                            output_sequence_length=MAX_OUTPUT_LENGTH)  # desired output length of vectorized sequences
    text_vectorizer.adapt(train_sentences)
    return text_vectorizer


def Create_Embedding_Layer(input_dim):
    print("input_dim: ", input_dim)
    embedding = layers.Embedding(input_dim=input_dim,  # length of vocabulary
                               output_dim=128,  # Note: different embedding sizes result in drastically different numbers of parameters to train
                               mask_zero=True,  # Use masking to handle variable sequence lengths (save space)
                               name="token_embedding")
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
        # TF-IDF, Multinomial Naive Bayes
        # Create tokenization and modelling pipeline
        model = Pipeline([
                        ("tf-idf", TfidfVectorizer()),
                        ("clf", MultinomialNB())
                        ])
        return model

    def Model_1(self, train_sentences, num_classes):
        """
        Create and return a 1D Convolutional model
        """
        text_vectorizer = Create_Text_Vectorizer(train_sentences)

        inputs = layers.Input(shape=(1,), dtype=tf.string)
        x = text_vectorizer(inputs) # vectorize text inputs
        x = Create_Embedding_Layer(len(text_vectorizer.get_vocabulary()))(x)    # create embedding
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)  # condense the output of our feature vector
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        # Compile model
        model.compile(loss="categorical_crossentropy",  # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
        return model