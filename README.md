# Amazon_Review_Sentiment_Analysis

## Abstract
- A sentiment analysis model to classify Amazon product reviews as positive or negative.
- 

## How to use
* Running system example:
  * WSL2 / Ubuntu : 22.04
  * TensorFlow : 2.14.0
  * CUDA : 11.8
  * cuDNN : 8.7


https://amazon-reviews-2023.github.io/
Used wsl2 to run tensorflow.
https://hsleeword.wordpress.com/category/tech/%EC%86%8C%EC%86%8C%ED%95%9C-tip/

https://dsaint31.tistory.com/328
# Data Overview
Source[https://amazon-reviews-2023.github.io/]: Amazon Reviews (May 1996 - Oct 2018)
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
Since the data is written in Json format, I loaded and converted it to TFRecord(a format using memory efficently) that I can use it for training. At the same time I preprocessed the data.

## Load and preprocessing
* Convert two Json files to data frames with 'reviewText', 'overall', and 'vote' columns.
* Apply preprocess function I created
* Sampling the data
  * Sample size = 100000 for each dataframes (Due to physical memory issue)
  * weight value of 'vote' ( more votes, more reliability)
  * drop 'vote' column since I do not need it anymore
* return df

## Positivity
* newly created column
* if overall rating is over 3, label as 1, otherwise 0
 * drop 'overall' column since I do not need it anymore

## TFRecord
* Transform the dataframe to TFRecord format.
* Requires serialization method to do so.

# training.py
With preprocessed TFRecord data of 'reviewText' and 'positivity' column, I converted it data frame for the trainig. And then split those data and processed Tokenizer and Padding. After all train data and tokenizer that will be used for modeling part is ready, return it.

## Processing Data
* Load the data and convert it the dataframe
* Setting X_df as 'reviewText' and y_df as 'positivity' and split train and test data
* Porcess Tokenizer and Padding toward train and test data for 'reviewText'
  * Using tensorflow.keras
  * Save the tokenizer as Json format for the prediction model.
* Return X_train, X_test, y_train, y_test and tokenizer


# modeling.py
Using TensorFlow, I built a model that can determine whether the text is negative or positive

## TensorFlow Model
* get X_train, X_test, y_train, y_test, tokenizer using process_data function from training.py
* 



# Data Processing
## Spliting test set and train set


label 1 if over 3 otherwise 0

Let's check if there are any duplicate

(5 11374381 2)
Since the total number of review was 13638545, but without any duplicate, it's 11374381. We have to remove those duplicate data.
Total number of reviews : 11374382

print(df.isnull().values.any()) 
True

Total number of reviews : 1781497

Number of train sample : 1336122
Number of test sample : 445375


## Data Cleaning

With regular expression let's find the review only with English. 
Number of train sample after data cleaning : 1336122
Number of test sample after data cleaning : 445375
All English check!

## Tokenization


## Integer Encoding

Now, I want to remove least repeated word from the train set. It may be helpful to remove words that are repeated less than 2 times.
I set the threshold as 2 and counted the repeated number and compare.
Size of vocabulary set: 749175
Number of unnecessary words that is repeated less than 1 times: 448230
Percentage of unnecessary words: 59.829812794073476
Percentage of frequency of unnecessary words: 0.3787618369094778
