import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
senteces = ["I am Neel Parikh","He is Karan Dave", "I am Good Human"]
tokenizer = Tokenizer(num_words = 100, oov_token = "<NNP>")
tokenizer.fit_on_texts(senteces)
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(senteces)
print(sequences)

padded = pad_sequences(sequences, maxlen = 10)
print(padded)

print("\Testing.......")
testing = ["I am Bin Parikh", "He loves Karan Dave"]
tokenizer = Tokenizer(num_words = 100, oov_token = "<KKP>")
tokenizer.fit_on_texts(testing)
word_index = tokenizer.word_index
print(word_index)
sequences_test = tokenizer.texts_to_sequences(testing)
print(sequences_test)
padded = pad_sequences(sequences_test,maxlen=6)
print(padded)