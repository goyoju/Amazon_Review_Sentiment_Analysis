import tensorflow as tf
import pandas as pd
import json
import re
import numpy as np
from sklearn.model_selection import train_test_split

def serialization(reviewText, positivity):
 """
 Takes reviewText and positivity value and serializes them
 """
 feature = {
  'reviewText': tf.train.Feature(bytes_list=tf.train.BytesList(value=[reviewText.encode('utf-8')])),
  'positivity': tf.train.Feature(int64_list=tf.train.Int64List(value=[positivity]))
 }

 proto = tf.train.Example(features=tf.train.Features(feature=feature))    # creating a Features message using tf.train.Example.


 return proto.SerializeToString()    # return serialized string

def df_to_tfrecord(df, f):
 """
 Saves the dataframe in TFRecord format.
 """
 with tf.io.TFRecordWriter(f) as writer:
  for index, row in df.iterrows():
   serialized_df = serialization(row['reviewText'], row['positivity'])     # Serialization
   writer.write(serialized_df)

def preprocess_text(t):
 """
 preprocess the text
 """
 if pd.isna(t):
  return ''
 t = t.lower()   #lower case
 t = re.sub(r'[^a-zA-Z\s]', '', t)   # using regular expression to remove everything but English
 return t

def load_and_preprocess_data(f, sample_size):
 """
 loads data and returns preprocessed data 
 """

 data = []
    
 # open Json file and write the dataframe line by line
 with open(f, 'r', encoding='utf-8') as file:
     for line in file:
         json_obj = json.loads(line)
         Json_data = {
             'reviewText': json_obj.get('reviewText', ''),
             'overall': json_obj.get('overall', 0),
             'vote': json_obj.get('vote', '0')
         }
         data.append(Json_data)
    
 # creating a dataframe with imported data
 df = pd.DataFrame(data)
    
 # preprocessing the reviewText
 df['reviewText'] = df['reviewText'].apply(preprocess_text)
    
 # Removing Nan values and changing to numerical values for vote
 df['vote'] = pd.to_numeric(df['vote'], errors='coerce').fillna(0)
 
 # sampling the data with defined sample size
 df = df.sample(n=sample_size, weights='vote', random_state=42)   # vote is the weight value for accuracy improvement

 df.drop(columns=['vote'], inplace=True)   # dropping vote column since we do not need it anymore

 return df


if __name__ == "__main__":
 # importing electronics reviews and Home and Kitchen reviews data
 electronics = 'data/Electronics_5.json'
 home_kitchen = 'data/Home_and_Kitchen_5.json'

 # loading and preprocess the data
 sample_size = 100000 # depends on the system capability  
 electronics_reviews = load_and_preprocess_data(electronics, sample_size)
 home_kitchen_reviews = load_and_preprocess_data(home_kitchen, sample_size)

 # Combining electronics review and Home and Kitchen review data
 combined_reviews = pd.concat([electronics_reviews, home_kitchen_reviews], ignore_index=True)


 combined_reviews.drop_duplicates(subset=['reviewText'], inplace=True)   #dropping all the duplicate data
 combined_reviews.dropna(inplace=True)    #dropping Nan value

 # 
 combined_reviews['positivity'] = combined_reviews['overall'].apply(lambda x: 1 if x > 3 else 0)

 """
 Optional : Calculating the ratio of the positivity
 positive_samples = combined_reviews['positivity'].sum()
 negative_samples = combined_reviews.shape[0] - positive_samples

 ratio = positive_samples / negative_samples

 print(f"positive / negative sample ratio: {ratio:.3f}")
 combined_reviews.drop(columns=['overall'], inplace=True) #drop overall
 """

 combined_reviews.drop(columns=['overall'], inplace=True) #drop overall

 tfrecord_filename = 'reviews.tfrecord'
 df_to_tfrecord(combined_reviews, tfrecord_filename)

 print(f"Data has been successfully converted to {tfrecord_filename}.")