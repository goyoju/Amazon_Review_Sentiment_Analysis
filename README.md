# Amazon_Review_Sentiment_Analysis


## Abstract
This project aims to perform sentiment analysis on Amazon reviews ranging from May 1996 to October 2018. By analyzing customer reviews in the Electronics and Home & Kitchen categories, the project predicts whether a review is positive or negative.

## How to use
* Running system example:
  * WSL2 / Ubuntu 22.04
  * Python 3
  * TensorFlow 2.14.0
  * CUDA : 11.8
  * cuDNN : 8.7
  * Hardware:
   * GPU : RTX 4060TI 6GB
   * Memory : 32 GB

# Dataset download
First, get the git first.download Run this on the terminal.
'''
$ git pull
'''
Then download the dataset from DVC.
'''
$ dvc pull
'''

### Setting Up:
'''
$ python3 setting_up.py
'''

It will autometically train the data and create model.

### Prediction model:
'''
$ python3 prediction.py
'''
Then you can type anything to predict whether negative or positive
Ex.


# Data Overview
Source: (Amazon Reviews (May 1996 - Oct 2018))[https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/]
Format: JSON
Datasets for the prediction model : **Electronics_5.json**, **Home_and_Kitchen_5.json**
Example data: 
{
  "sort_timestamp": 1634275259292,
  "rating": 3.0,
  "helpful_votes": 0,
  "title": "Meh",
  "text": "These were lightweight and soft but much too small for my liking. I would have preferred two of these together to make one loc. For that reason I will not be repurchasing.",
  "images": [
    {
      "small_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL256_.jpg",
      "medium_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL800_.jpg",
      "large_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL1600_.jpg",
      "attachment_type": "IMAGE"
    }
  ],
  "asin": "B088SZDGXG",
  "verified_purchase": true,
  "parent_asin": "B08BBQ29N5",
  "user_id": "AEYORY2AVPMCPDV57CE337YU5LXA"
}

Total number of reviews : 13638545

# TFRecord_convert.py
The raw data, formatted in JSON, is loaded and converted into a more memory-efficient format, TFRecord, for training purposes. The preprocessing steps include:

## Load and preprocessing
* Converting JSON files to data frames with columns: reviewText, overall, and vote.
* Applying custom preprocessing functions.
* Sampling the data
  * Sampling data due to physical memory limitations (sample size = 100,000 for each dataframe)
  * Assigning weight values based on vote counts for reliability
  * Dropping the vote column.

## Positivity
* Creating a positivity column where ratings over 3 are labeled as 1 (positive), otherwise 0 (negative)
* ㅇropping the overall column.

## TFRecord
* The data is then transformed into TFRecord format, requiring serialization methods.

# training_setting.py
The preprocessed TFRecord data, consisting of reviewText and positivity, is loaded, converted into data frames, and prepared for training.

## Processing Data
* Loading data and converting it to a dataframe.
* Setting X_df as reviewText and y_df as positivity, and splitting into train and test datasets.
* Processing Tokenization and Padding for the reviewText data
  * Using tensorflow.keras
* Saving the tokenizer in JSON format for the prediction model.
* Returning X_train, X_test, y_train, y_test, and the tokenizer.


# modeling.py
Using TensorFlow, I built a model that can determine whether the text is negative or positive

## TensorFlow Model
* Using the TensorBoard callback to monitor the training process.
* Unlocking the GPU memory usage, especially beneficial for systems running WSL2.
* Obtaining X_train, X_test, y_train, y_test, and the tokenizer via the process_data function from training.py.
* Adjusting Hyperparameters:
 * Tuning the model with specific hyperparameters including Embedding Dimension, Neuron Units, Vocabulary Size, Batch Size, and Number of Epochs.
* Constructing the Model:
 * Adding a Dense layer with a sigmoid activation function suitable for binary classification.
* Implementing EarlyStopping and ModelCheckpoint callbacks for improved training and memory efficiency.
* Loading and saving the model data in HDF5 format for efficient storage and access.
* Prediction:
 * Assessing test accuracy to validate the model’s predictive capability.

# sentiment_predict.py
* Applying preprocessing to the user-provided string using the tokenizer from training.py.
* Utilizing the trained model to predict whether the input string conveys a negative or positive sentiment.
