# Amazon Reviews Sentiment Analysis

## Abstract
This project aims to perform sentiment analysis on Amazon reviews ranging from May 1996 to October 2018. By analyzing customer reviews in the Electronics and Home & Kitchen categories, the project predicts whether a review is positive or negative.

Ex.


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

### Dataset download
First, update your local repository:

{

    "$ git pull",
}



Next, download the dataset with DVC:

{

    $ dvc pull
}



### Setting Up:
Run the setup script to prepare your environment automatically:

{

    $ python3 setting_up.py
}

### Prediction model:
To predict whether a given input text is negative or positive, run the prediction script:

{

    $ python3 prediction.py
}

Input any text when prompted to receive a sentiment prediction.


# Data Overview
Source: **[Amazon Reviews (May 1996 - Oct 2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)**

Format: **JSON**

Datasets for the prediction model : **Electronics_5.json**, **Home_and_Kitchen_5.json**

**Example data:**

{

  "sort_timestamp": 1634275259292,

  "rating": 3.0,

  "helpful_votes": 0,
 
  "title": "Meh",
 
  "text": "These were lightweight and soft but much too small for my liking. I would have preferred two of these together to make one loc. For that reason I will not be repurchasing.",

  "images": [{
       "small_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL256_.jpg",
       "medium_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL800_.jpg",
       "large_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL1600_.jpg",
       "attachment_type": "IMAGE"
  }],

  "asin": "B088SZDGXG",

  "verified_purchase": true,

  "parent_asin": "B08BBQ29N5",

  "user_id": "AEYORY2AVPMCPDV57CE337YU5LXA"

}

**Total number of reviews : 13,638,545**

# TFRecord_convert.py
Preprocessing steps include converting JSON files to dataframes, sampling, and transforming into TFRecord format for efficient training.

## Load and preprocessing
* Convert JSON files to data frames with columns: reviewText, overall, and vote.
* Apply custom preprocessing functions.
* Sampling the data
  * Sample data due to physical memory limitations (sample size = 100,000 for each dataframe)
  * Assign weight values based on vote counts for reliability
  * Drop the vote column.

## Positivity
* Create a positivity column where ratings over 3 are labeled as 1 (positive), otherwise 0 (negative)
* Drop the overall column.

## TFRecord
* The data is then transformed into TFRecord format, requiring serialization methods.

# training_setting.py
The preprocessed data is loaded, processed for tokenization and padding, and prepared for model training.

## Processing Data
* Load and convert data into a dataframe.
* Set X_df as reviewText and y_df as positivity, then split.
* Process Tokenization and Padding for the reviewText data
  * Use tensorflow.keras
* Save the tokenizer in JSON format for the prediction model.
* Returning X_train, X_test, y_train, y_test, and the tokenizer.


# modeling.py
A TensorFlow model is constructed to classify texts based on sentiment.

## TensorFlow Model
* Use the TensorBoard callback to monitor the training process.
* Adjust for WSL2 GPU memory usage.
* Obtain X_train, X_test, y_train, y_test, and the tokenizer via the process_data function from training.py.
* Adjust Hyperparameters:
  * Tune the model with specific hyperparameters including Embedding Dimension, Neuron Units, Vocabulary Size, Batch Size, and Number of Epochs.
* Construct the Model:
  * Add a Dense layer with a sigmoid activation function suitable for binary classification.
* Implement EarlyStopping and ModelCheckpoint callbacks for improved training and memory efficiency.
* Load and save the model data in HDF5 format for efficient storage and access.
* Prediction:
  * Assess test accuracy to validate the model’s predictive capability.

# sentiment_predict.py
* Apply preprocessing to the user-provided string using the tokenizer from training.py.
* Utilize the trained model to predict whether the input string conveys a negative or positive sentiment.
