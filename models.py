"""
Models
"""
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

MAX_VOCAB_LENGTH = 68000  # Mentioned in Table 2 in https://arxiv.org/abs/1710.06071
MAX_OUTPUT_LENGTH = 55  # covers 95 percentile of sequence lengths in dataset

alphabet = string.ascii_lowercase + string.digits + string.punctuation
NUM_CHAR_TOKENS = len(alphabet) + 2


def Create_Char_Text_Vectorizer(train_chars, output_seq_char_len):
    text_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,  
                            output_sequence_length=output_seq_char_len,
                            standardize="lower_and_strip_punctuation",
                            name="char_vectorizer")
    text_vectorizer.adapt(train_chars)
    return text_vectorizer


def Create_Token_Text_Vectorizer(train_sentences):
    text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_LENGTH,  # number of words in vocabulary
                            output_sequence_length=MAX_OUTPUT_LENGTH)  # desired output length of vectorized sequences
    text_vectorizer.adapt(train_sentences)
    return text_vectorizer


def Create_Embedding_Layer(input_dim, output_dim):
    embedding = layers.Embedding(input_dim=input_dim,  # length of vocabulary
                            output_dim=output_dim,  # Note: different embedding sizes result in drastically different numbers of parameters to train
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
        Create and return a 1D Convolutional model with token level vectorizer
        """
        text_vectorizer = Create_Token_Text_Vectorizer(train_sentences)

        inputs = layers.Input(shape=(1,), dtype=tf.string)
        x = text_vectorizer(inputs) # vectorize text inputs
        x = Create_Embedding_Layer(len(text_vectorizer.get_vocabulary()), 128)(x)    # create embedding
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)  # condense the output of our feature vector
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        # Compile model
        model.compile(loss="categorical_crossentropy",  # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model

    def Model_2(self, num_classes):
        """
        Create and return a transfer learning model
        """
        import tensorflow_hub as hub

        sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[], # shape of inputs coming to our model 
                                        dtype=tf.string, # data type of inputs coming to the USE layer
                                        trainable=False, # keep the pretrained weights (we'll create a feature extractor)
                                        name="Universal_Sentence_Encoder")
        model = tf.keras.Sequential([
                        sentence_encoder_layer, # take in sentences and then encode them into an embedding
                        layers.Dense(128, activation="relu"),
                        layers.Dense(num_classes, activation="softmax")
                        ], name="model_2_USE")

        # Compile model
        model.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model

    def Model_3(self, train_chars, output_seq_char_len, num_classes):
        """
        Create and return a 1D Convolutional model with character level vectorizer
        """
        text_vectorizer = Create_Char_Text_Vectorizer(train_chars, output_seq_char_len)

        inputs = layers.Input(shape=(1,), dtype="string")
        x = text_vectorizer(inputs)
        x = Create_Embedding_Layer(len(text_vectorizer.get_vocabulary()), 25)(x)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.GlobalMaxPool1D()(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs,
                        outputs=outputs,
                        name="model_3_conv1D_char_embedding")

        # Compile model
        model.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model
