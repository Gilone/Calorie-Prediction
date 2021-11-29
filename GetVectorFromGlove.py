# from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy as np

# def process_glove_data():
#     glove_input_file = '.\glove.840B.300d.txt'
#     word2vec_output_file = '.\glove.840B.300d.word2vec.txt'
#     (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
#     print(count, '\n', dimensions)

class GloveModel:
    def __init__(self, glove_path):
        word2vec_output_file = glove_path
        self.glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False, no_header=True)

    def get_vector(self, word):
        try:
            return self.glove_model[word]
        except KeyError:
            return np.random.rand(self.glove_model.vector_size)

    def get_similar_vector(self, word):
        return self.glove_model.most_similar(word)

    def get_average_vector_of_word_list(self, word_list):
        vector_list = []
        for w in word_list:
            vector_list.append(self.get_vector(w))
        return np.mean(vector_list, axis=0)

    def get_vector_seq_from_word_seq(self, word_seq):
        vector_seq = []
        for w in word_seq.split(' '):
            vector_seq.append(self.get_vector(w))
        return vector_seq

    def get_graph_of_a_word_list(self, word_list):
        vector_matrix = np.matrix([self.get_vector(word) for word in word_list])
        U, _, _ = la.svd(vector_matrix, full_matrices=False)
        for i in range(vector_matrix.shape[0]):  
            plt.text(U[i, 0], U[i, 1], word_list[i])
        coord = U[:, 0:2]
        plt.xlim((np.min(coord[:, 0]) - 0.1, np.max(coord[:, 0]) + 0.1))
        plt.ylim((np.min(coord[:, 1]) - 0.1, np.max(coord[:, 1]) + 0.1))
        plt.show()

'''EXAMPLE
GM = GloveModel('.\glove.840B.300d.txt') # download from https://nlp.stanford.edu/projects/glove/
print('milk vector: ', GM.get_vector('milk'))
print('butter vector: ', GM.get_vector('butter'))
print('water vector: ', GM.get_vector('water'))
print('average vector: ', GM.get_average_vector_of_word_list(['milk', 'butter', 'water']))
GM.get_graph_of_a_word_list(['milk', 'butter', 'water'])
'''

