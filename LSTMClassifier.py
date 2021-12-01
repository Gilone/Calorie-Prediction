from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from GetVectorFromGlove import GloveModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow.keras.backend as K
import tensorflow_addons as tfa 

class StepsClassifier():

    def __init__(self, train_seq, training_label_seq, validation_seq, validation_label_seq, glove_model=None, activation='relu', max_length = 200, trainable=True):
        self.train_seq = train_seq
        self.activation = activation
        self.training_label_seq = training_label_seq
        self.validation_seq = validation_seq
        self.validation_label_seq = validation_label_seq
        self.embedding_dim = 300
        self.trainable = trainable
        self.max_length = max_length
        self.labels = ['0', '1', '2']
        if not glove_model:
            self.GM = GloveModel('.\glove.840B.300d.txt')
        else:
            self.GM = glove_model
        self._inital_token()
        self._inital_model()


    def _inital_token(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.train_seq)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index)+1
        self.emb_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, index in self.word_index.items():
            self.emb_matrix[index, :] = self.GM.get_vector(word) # potential bug

    # def _recall_m(self, y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     return recall

    # def _precision_m(self, y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     return precision


    # def _f1_m(self, y_true, y_pred):
    #     precision = self._precision_m(y_true, y_pred)
    #     recall = self._recall_m(y_true, y_pred)
    #     return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def _inital_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim, input_length=self.max_length, \
                trainable=self.trainable), # weights = [self.emb_matrix], 
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.embedding_dim)),
            tf.keras.layers.Dense(self.embedding_dim, activation=self.activation),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, average='macro')])

    def _plot_training_graphs(self, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    def paddle_all_seq(self, steps_string_list):
        train_token_sequences = self.tokenizer.texts_to_sequences(steps_string_list)
        train_padded = pad_sequences(train_token_sequences, maxlen=self.max_length, padding='post', truncating='post')
        return train_padded

    def train(self, num_epochs=10):
        train_padded = self.paddle_all_seq(self.train_seq)
        valid_padded = self.paddle_all_seq(self.validation_seq)
        history = self.model.fit(train_padded, self.training_label_seq, epochs=num_epochs, validation_data=(valid_padded, self.validation_label_seq), verbose=2)
        self._plot_training_graphs(history, "accuracy")
        self._plot_training_graphs(history, "loss")

    def _predict_single(self, test_seq):
        test_padded = self.paddle_all_seq([test_seq])
        return self.model.predict(test_padded)

    def prediction_single(self, test_seq):
        pred = self._predict_single(test_seq)
        return self.labels[np.argmax(pred[-1])]

    def evaluate_model(self, test_seq=None, test_label_seq=None):
        if not test_seq or test_label_seq:
            test_seq = self.validation_seq
            test_label_seq = self.validation_label_seq
        test_seq = self.paddle_all_seq(test_seq)
        loss,  accuracy, f1 = self.model.evaluate(test_seq, test_label_seq, verbose=1) 
        return accuracy, f1