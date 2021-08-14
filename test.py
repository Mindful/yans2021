from data.db import DbConnection
from nlp.embedding import sort_words_by_distance

a = DbConnection('css')
r = a.get_clusters_for_lemma('play')
z = a.get_clusters_for_lemma('dog')


def cluster_data_for_lemma(lemma: str, display_limit: int = 50):

    #TODO: conflating clusters and parts of speech - pick one POS
    raw_data = a.get_clusters_for_lemma(lemma)
    for word_cluster, word_list in raw_data:
        words_by_cluster_label = {
            label: sort_words_by_distance([word for word, word_label in zip(word_list, word_cluster.labels)
                                           if word_label == label], centroid)
            for label, centroid in zip(sorted(set(word_cluster.labels)), word_cluster.cluster_centers)
        }

        display_data = {'clusters': [
            {
                'name': f'{lemma} cluster {label}',
                'data': [
                            {'x': word.display_embedding[0],
                             'y': word.display_embedding[1],
                             'z': word.display_embedding[2],
                             'text': word.sentence} for word in cluster_words[:display_limit]
                        ]
            } for label, cluster_words in words_by_cluster_label.items()
        ]}




print('debuggy')