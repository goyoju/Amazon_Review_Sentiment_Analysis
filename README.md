# Amazon_Review_Sentiment_Analysis

 https://jmcauley.ucsd.edu/data/amazon/
Used wsl2 to run tensorflow.
https://hsleeword.wordpress.com/category/tech/%EC%86%8C%EC%86%8C%ED%95%9C-tip/

https://dsaint31.tistory.com/328
# Data Overview
Total number of reviews : 13638545
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
using spacy
pip install spacy
python -m spacy download en_core_web_sm

Also use stopwords in spaCy to remove unnecessary tokens.

## Integer Encoding

Now, I want to remove least repeated word from the train set. It may be helpful to remove words that are repeated less than 2 times.
I set the threshold as 2 and counted the repeated number and compare.
Size of vocabulary set: 749175
Number of unnecessary words that is repeated less than 1 times: 448230
Percentage of unnecessary words: 59.829812794073476
Percentage of frequency of unnecessary words: 0.3787618369094778
