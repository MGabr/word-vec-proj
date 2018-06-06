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
    "arbeitslos",
    "reich"
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
    "gene",
    "beine",
    "herz",
    "lunge",
    "bakterien"
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
    "arm",
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

print(normal_window_model.wv.most_similar(positive=["leber", "herz", "lunge"], topn=5))
print(small_window_model.wv.most_similar(positive=["leber", "herz", "lunge"], topn=5))
# [('niere', 0.8039090633392334), ('nieren', 0.8003708124160767), ('magen', 0.78350430727005), ('milz', 0.7786546945571899), ('hirn', 0.776465117931366)]
# [('niere', 0.7802958488464355), ('nieren', 0.7620891332626343), ('gebärmutter', 0.7612006068229675), ('milz', 0.7540740966796875), ('schilddrüse', 0.7426707744598389)]

print(normal_window_model.wv.most_similar(positive=["arm", "bein"], topn=5))
print(small_window_model.wv.most_similar(positive=["arm", "bein"], topn=5))
# [('unterarm', 0.8634679317474365), ('handgelenk', 0.8253477811813354), ('unterschenkel', 0.8162782192230225), ('oberschenkel', 0.8147163391113281), ('ellbogen', 0.8083438873291016)]
# [('unterarm', 0.8030626773834229), ('handgelenk', 0.7994979619979858), ('oberarm', 0.7810859680175781), ('oberschenkel', 0.7748416066169739), ('knöchel', 0.7628743052482605)]

print(normal_window_model.wv.most_similar(positive=["lebendig", "arm"], negative=["reich"], topn=5))
print(small_window_model.wv.most_similar(positive=["lebendig", "arm"], negative=["reich"], topn=5))
# [('leblos', 0.6285403966903687), ('versteinert', 0.6115909814834595), ('verkrüppelt', 0.6076324582099915), ('fürchterlich', 0.6031007766723633), ('geistesabwesend', 0.5885956883430481)]
# [('leblos', 0.5918615460395813), ('verkrüppelt', 0.5835748314857483), ('gefesselt', 0.5736857652664185), ('sehend', 0.5733113288879395), ('oberkörper', 0.5618880391120911)]

print(normal_window_model.similarity("arm", "bein"))
print(normal_window_model.similarity("arm", "reich"))
print(normal_window_model.similarity("arm", "besitzlos"))
# 0.6806009454634425
# 0.10973013916901389
# 0.14481998860155842

print(small_window_model.similarity("arm", "bein"))
print(small_window_model.similarity("arm", "reich"))
print(small_window_model.similarity("arm", "besitzlos"))
# 0.6406667898287061
# 0.25179555028233397
# 0.1754358445531931

print(normal_window_model.similarity("erben", "geld"))
print(normal_window_model.similarity("erben", "gene"))
print(normal_window_model.similarity("vererben", "geld"))
print(normal_window_model.similarity("vererben", "gene"))
# 0.37418759210127833
# 0.07244948610288553
# 0.3419726098723458
# 0.1395088873826763

print(small_window_model.similarity("erben", "geld"))
print(small_window_model.similarity("erben", "gene"))
print(small_window_model.similarity("vererben", "geld"))
print(small_window_model.similarity("vererben", "gene"))
# 0.41059506747134494
# 0.10989564525191263
# 0.42627524233988384
# 0.18963249209746852

print(normal_window_model.wv.most_similar(positive=["euro", "dollar", "yen"], topn=5))
print(small_window_model.wv.most_similar(positive=["euro", "dollar", "yen"], topn=5))
# [('gbp', 0.7996298670768738), ('rupien', 0.7676671743392944), ('peseten', 0.7548888325691223), ('chf', 0.7522148489952087), ('eur', 0.7432374954223633)]
# [('peseten', 0.7984158992767334), ('gbp', 0.7947422862052917), ('rupien', 0.7789661288261414), ('rubel', 0.7781881093978882), ('pesos', 0.7672573328018188)]
