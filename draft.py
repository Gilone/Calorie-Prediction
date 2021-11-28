from GetVectorFromGlove import GloveModel

GM = GloveModel('.\glove.840B.300d.txt')

words_set = set()

def update_words_set(word_list):
    for word in word_list:
        words_set.add(word)

def get_ingredients_features(ingredients_list):
    words_list = []
    for ingre in ingredients_list:
        words_list.extend(ingre.split(' '))
    update_words_set(words_list)
    average_word_vector = GM.get_average_vector_of_word_list(words_list)
    return average_word_vector