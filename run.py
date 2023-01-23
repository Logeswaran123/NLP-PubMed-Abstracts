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
                                epochs=5,
                                validation_data=valid_dataset)
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
                                epochs=5,
                                validation_data=valid_dataset)
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
                                epochs=5,
                                validation_data=val_char_dataset)
    model_3.evaluate(val_char_dataset)

    # Predict on validation data and calculate scores
    model_3_pred_probs = model_3.predict(val_char_dataset)
    model_3_results = calculate_results(y_true=val_labels_encoded, y_pred=tf.argmax(model_3_pred_probs, axis=1))
    print("\n1D Convolutional model (char embeddings) Results:\n", model_3_results)
    print("\n-----------------------------------------------------\n")


    # Combine chars and tokens into a dataset
    train_char_token_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars))
    train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
    train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels))
    val_char_token_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars))
    val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
    val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))

    # Prefetch and batch train and val data
    train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # Create Hybrid model with token and character level embeddings
    model_4 = models.Model_4(train_sentences, train_chars, output_seq_char_len, num_classes)

    model_4_history = model_4.fit(train_char_token_dataset, # train on dataset of token and characters
                                epochs=5,
                                validation_data=val_char_token_dataset)
    model_4.evaluate(val_char_token_dataset)

    # Predict on validation data and calculate scores
    model_4_pred_probs = model_4.predict(val_char_token_dataset)
    model_4_results = calculate_results(y_true=val_labels_encoded, y_pred=tf.argmax(model_4_pred_probs, axis=1))
    print("\nHybrid model (token and char embeddings) Results:\n", model_4_results)
    print("\n-----------------------------------------------------\n")


    # Create one-hot-encoded tensors of our "line_number" column 
    train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
    val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
    test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)

    # Create one-hot-encoded tensors of our "total_lines" column 
    train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
    val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
    test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)

    # Create training and validation datasets (all four kinds of inputs)
    # Combine chars, tokens, line numbers, total lines into a dataset
    train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot, # line numbers
                                                                train_total_lines_one_hot, # total lines
                                                                train_sentences, # train tokens
                                                                train_chars)) # train chars
    train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # train labels
    train_pos_char_token_dataset = tf.data.Dataset.zip((train_pos_char_token_data, train_pos_char_token_labels)) # combine data and labels
    val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                                val_total_lines_one_hot,
                                                                val_sentences,
                                                                val_chars))
    val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
    val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))

    # Prefetch and batch train and val data
    train_pos_char_token_dataset = train_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    val_pos_char_token_dataset = val_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # Create Hybrid model with token, character and position embeddings
    model_5 = models.Model_5(train_sentences, train_chars, output_seq_char_len, num_classes)

    model_5_history = model_5.fit(train_pos_char_token_dataset,
                                epochs=5,
                                validation_data=val_pos_char_token_dataset)
    model_5.evaluate(val_pos_char_token_dataset)

    # Predict on validation data and calculate scores
    model_5_pred_probs = model_5.predict(val_pos_char_token_dataset)
    model_5_results = calculate_results(y_true=val_labels_encoded, y_pred=tf.argmax(model_5_pred_probs, axis=1))
    print("\nHybrid model (token, char and position embeddings) Results:\n", model_5_results)
    print("\n-----------------------------------------------------\n")