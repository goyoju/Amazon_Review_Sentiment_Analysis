import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

def sentiment_predict(sentence, tokenizer):
    # preprocess the new sentence
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence).lower()
    sequences = tokenizer.texts_to_sequences([sentence])
    pad = pad_sequences(sequences, maxlen=500)

    model_final = load_model('best_model.h5')

    # prediction
    score = float(model_final.predict(pad))

    # determining negative or positive
    if score > 0.5:
        return "{:.2f}% probability of being a positive review.".format(score * 100)
    else:
        return "{:.2f}% probability of being a negative review.".format((1 - score) * 100)

if __name__ == "__main__":
    # loading tokenizer data
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)

    while True:
        # Input from the user
        user_input = input("(Type quit to exit) Enter an example sentence!: ")
        
        # exit function
        if user_input.lower() == 'quit':
            print("Done")
            break

        # Result
        print(sentiment_predict(user_input, tokenizer))