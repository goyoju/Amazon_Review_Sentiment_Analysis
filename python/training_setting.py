import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def load_dataset(filename):
  """
  Load data from TFRecord
  """
  raw_dataset = tf.data.TFRecordDataset(filename)
  
  # taking components 
  feature = {
    'reviewText': tf.io.FixedLenFeature([], tf.string),
    'positivity': tf.io.FixedLenFeature([], tf.int64),
  }
  def _parse_function(p):
    return tf.io.parse_single_example(p, feature)


  parsed_dataset = raw_dataset.map(_parse_function)

  return parsed_dataset

def parsed_records_to_dataframe(dataset):
  """
  Converting to dataframe
  """
  reviewText, positivity = [], []
  for parsed_record in dataset:
    reviewText.append(parsed_record['reviewText'].numpy().decode('utf-8'))
    positivity.append(parsed_record['positivity'].numpy())
  df = pd.DataFrame({'reviewText': reviewText, 'positivity': positivity})
  return df

def create_dataset_from_tfrecord(tfrecord_file):
  # Creating dataset
  raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
  # Mapping the data
  parsed_dataset = raw_dataset.map(_parse_function)
  return parsed_dataset

def process_data(filename='reviews.tfrecord', test_size=0.25, max_len=500):  # taking reviews.tfrecord
  """
  Load the data and split
  """
  dataset = load_dataset(filename)
  df = parsed_records_to_dataframe(dataset)

  # X is reviewText, y is positivity  
  X_df = df['reviewText']
  y_df = df['positivity']

  # Spliting the data
  X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_size, random_state=42)

  #Tokenizing the reviewText
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_train)

  train_sequences = tokenizer.texts_to_sequences(X_train)
  test_sequences = tokenizer.texts_to_sequences(X_test)

  # Padding
  X_train = pad_sequences(train_sequences, maxlen=max_len)
  X_test = pad_sequences(test_sequences, maxlen=max_len)

  #saving the tokenizer into JSON format
  tokenizer_json = tokenizer.to_json()
  with open('tokenizer.json', 'w', encoding='utf-8') as f:
      f.write(json.dumps(tokenizer_json, ensure_ascii=False))

  return X_train, X_test, y_train.values, y_test.values, tokenizer

if __name__ == "__main__":
  X_train, X_test, y_train, y_test, tokenizer = process_data()

  print('Preprocessing done. Data is ready for training.')