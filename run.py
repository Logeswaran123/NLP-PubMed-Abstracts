import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf

from models import Model
from preprocess import load_and_preprocess
from utils import calculate_results, split_chars

SAVE_DIR = "model_logs"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", required=True, default="dataset",
                        help="Path to dataset dir", type=str)
    args = parser.parse_args()
    dataset_path = args.data

    # Preprocess and load data
    train_samples = load_and_preprocess(dataset_path + "train.txt")
    val_samples = load_and_preprocess(dataset_path + "dev.txt") # dev is another name for validation set (dev - development)
    test_samples = load_and_preprocess(dataset_path + "test.txt")
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)

    # Convert abstract text lines into lists
    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()
    test_sentences = test_df["text"].tolist()

    # Extract labels ("target" columns) and encode them into integers
    # LabelEncoder
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
    val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
    test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())
    num_classes = len(label_encoder.classes_)

    models = Model()

    # Create Baseline model
    baseline_model = models.Baseline()

    # Fit model on training data
    baseline_model.fit(train_sentences, train_labels_encoded)

    # Predict on validation data and calculate scores
    baseline_preds = baseline_model.predict(val_sentences)
    baseline_results = calculate_results(y_true=val_labels_encoded, y_pred=baseline_preds)
    print("\nBaseline model Results:\n", baseline_results)
    print("\n-----------------------------------------------------\n")


    # OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
    val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

    # Turn our data into TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
    valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))

    # Take the TensorSliceDataset's and turn them into prefetched batches
    # Refer: https://www.tensorflow.org/guide/data_performance#prefetching
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # Create 1D Convolutional model with token/word level embeddings
    model_1 = models.Model_1(train_sentences, num_classes)

    # Fit the model
    model_1_history = model_1.fit(train_dataset,
                                steps_per_epoch=int(0.1 * len(train_dataset)),  # only fit on 10% of batches for faster training time
                                epochs=5,
                                validation_data=valid_dataset,
                                validation_steps=int(0.1 * len(valid_dataset))) # only validate on 10% of batches
    model_1.evaluate(valid_dataset)

    # Predict on validation data and calculate scores
    model_1_pred_probs = model_1.predict(valid_dataset)
    model_1_results = calculate_results(y_true=val_labels_encoded, y_pred=tf.argmax(model_1_pred_probs, axis=1))
    print("\n1D Convolutional model (token embeddings) Results:\n", model_1_results)
    print("\n-----------------------------------------------------\n")


    # Create model using transfer learning
    model_2 = models.Model_2(num_classes)

    # Fit the model
    model_2_history = model_2.fit(train_dataset,
                                steps_per_epoch=int(0.1 * len(train_dataset)),  # only fit on 10% of batches for faster training time
                                epochs=5,
                                validation_data=valid_dataset,
                                validation_steps=int(0.1 * len(valid_dataset))) # only validate on 10% of batches
    model_2.evaluate(valid_dataset)

    # Predict on validation data and calculate scores
    model_2_pred_probs = model_2.predict(valid_dataset)
    model_2_results = calculate_results(y_true=val_labels_encoded, y_pred=tf.argmax(model_2_pred_probs, axis=1))
    print("\nTransfer learning model Results:\n", model_2_results)
    print("\n-----------------------------------------------------\n")


    # Split sequence-level data splits into character-level data
    train_chars = [split_chars(sentence) for sentence in train_sentences]
    val_chars = [split_chars(sentence) for sentence in val_sentences]
    test_chars = [split_chars(sentence) for sentence in test_sentences]

    # Find what character length covers 95% of sequences
    char_lens = [len(sentence) for sentence in train_sentences]
    output_seq_char_len = int(np.percentile(char_lens, 95))

    # Create char datasets
    train_char_dataset = tf.data.Dataset.from_tensor_slices((train_chars, train_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_char_dataset = tf.data.Dataset.from_tensor_slices((val_chars, val_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)

    # Create 1D Convolutional model with character level embeddings
    model_3 = models.Model_3(train_chars, output_seq_char_len, num_classes)

    # Fit the model
    model_3_history = model_3.fit(train_char_dataset,
                                steps_per_epoch=int(0.1 * len(train_char_dataset)),
                                epochs=5,
                                validation_data=val_char_dataset,
                                validation_steps=int(0.1 * len(val_char_dataset)))
    model_3.evaluate(val_char_dataset)

    # Predict on validation data and calculate scores
    model_3_pred_probs = model_3.predict(val_char_dataset)
    model_3_results = calculate_results(y_true=val_labels_encoded, y_pred=tf.argmax(model_3_pred_probs, axis=1))
    print("\n1D Convolutional model (char embeddings) Results:\n", model_3_results)
    print("\n-----------------------------------------------------\n")