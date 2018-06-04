from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# words to plot
economy_words = [
    "bank",
    "wirtschaft",
    "geld",
    "euro",
    "dollar",
    "inflation",
    "arbeitslos"
]

physics_words = [
    "elektrizität",
    "gravitation",
    "quanten",
    "atome",
    "physik"
]

biology_words = [
    "biologie",
    "tiere",
    "füße",
    "herz",
    "leber"
]

naturalscience_words = physics_words + biology_words

combined_words = [
    "gehalt",  # Einkommen vs Wert
    "länge",  # Ort vs Zeit
    "leistung",  # Arbeitsleistung vs Leistung (Physik)
    "materie",  # Thema vs Materie (Physik)
    "grenze",  # zwischen Ländern vs für Werte
    "organ",  # Institut vs Organ (Biologie)
    "arme",  # arme Personen vs Arme einer Person (Biologie)
    "erben",  # Geld erben vs DNA erben (Biologie)
]

words = economy_words + naturalscience_words + combined_words


def plot_words(model, save_plot_filename=None):
    # get word vectors for words to plot
    vecs = model[words]

    # perform PCA
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(vecs)

    # plot words
    len_ec = len(economy_words)
    pyplot.scatter(pca_vecs[:len_ec, 0], pca_vecs[:len_ec, 1], c="blue")

    len_ec_phy = len_ec + len(physics_words)
    pyplot.scatter(pca_vecs[len_ec:len_ec_phy, 0], pca_vecs[len_ec:len_ec_phy, 1], c="red")

    len_ec_phy_bio = len_ec_phy + len(biology_words)
    pyplot.scatter(pca_vecs[len_ec_phy:len_ec_phy_bio, 0],pca_vecs[len_ec_phy:len_ec_phy_bio, 1], c="yellow")

    pyplot.scatter(pca_vecs[len_ec_phy_bio:, 0],pca_vecs[len_ec_phy_bio:, 1], c="gray")

    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(pca_vecs[i, 0], pca_vecs[i, 1]))

    # save or show plot
    if save_plot_filename:
        pyplot.savefig(save_plot_filename)
    else:
        pyplot.show()
    pyplot.clf()


normal_window_model = Word2Vec.load("normal_window_model")
small_window_model = Word2Vec.load("small_window_model")

plot_words(normal_window_model, "figs/normal_window_economy_naturalscience.png")
plot_words(small_window_model, "figs/small_window_economy_naturalscience.png")

print(normal_window_model.wv.most_similar(positive=["wirtschaft", "organ" ], topn=5))
print(small_window_model.wv.most_similar(positive=["wirtschaft", "organ" ], topn=5))
# [('verbandsorgan', 0.6706839799880981), ('unternehmertum', 0.6644313335418701), ('finanzwesen', 0.6638613939285278), ('agrarpolitik', 0.625726044178009), ('fachausschuss', 0.62508225440979)]
# [('finanzwesen', 0.6784769296646118), ('frauenpolitik', 0.6730645895004272), ('bildungswesen', 0.6640677452087402), ('bildungspolitik', 0.6526181697845459), ('unternehmertum', 0.6525093913078308)]

print(normal_window_model.wv.most_similar(positive=["biologie", "organ" ], topn=5))
print(small_window_model.wv.most_similar(positive=["biologie", "organ" ], topn=5))
# [('genetik', 0.7177324295043945), ('entomologie', 0.7023565173149109), ('virologie', 0.702241837978363), ('lehrbuch', 0.6861299276351929), ('mykologie', 0.6850222945213318)]
# [('sportpädagogik', 0.6924142241477966), ('sozialmedizin', 0.6815182566642761), ('mykologie', 0.6803916692733765), ('sozialhygiene', 0.6773288249969482), ('psychologie', 0.6761468648910522)]
