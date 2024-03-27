from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from training_setting import process_data
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


# Make sure GPU is using all the memory resources
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)



# Takes train, test and tokenizer values  
X_train, X_test, y_train, y_test, tokenizer = process_data()

# Hyper parameter
embedding_dim = 2500
cell_num_units = 64
vocab_size = 500

tf.debugging.set_log_device_placement(True)

log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)   #TensorBoard to track the training


# Allocate the model only to GPU
with tf.device('/GPU:0'):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GRU(cell_num_units))
    model.add(Dense(1, activation='sigmoid'))



    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)   # saving only best model

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=2, callbacks=[es, mc, tensorboard_callback], batch_size=256, validation_split=0.2)


    model_final = load_model('best_model.h5')  # Taking the best model

    print("\n Test Accuracy : %.4f" % (model_final.evaluate(X_test, y_test)[1]))  # Accuracy