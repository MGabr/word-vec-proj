from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec


wiki_corpus = WikiCorpus("dewiki-latest-pages-articles.xml.bz2", dictionary={None: None})

normal_window_model = Word2Vec(window=5)
normal_window_model.build_vocab(wiki_corpus.get_texts())
normal_window_model.train(wiki_corpus.get_texts(),
                          total_examples=normal_window_model.corpus_count,
                          epochs=normal_window_model.epochs)
normal_window_model.save("normal_window_model")

small_window_model = Word2Vec(window=2)
small_window_model.build_vocab(wiki_corpus.get_texts())
small_window_model.train(wiki_corpus.get_texts(),
                         total_examples=small_window_model.corpus_count,
                         epochs=small_window_model.epochs)
small_window_model.save("small_window_model")

