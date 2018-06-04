from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# words to plot
countries = ["österreich", "deutschland", "frankreich", "spanien", "großbritannien", "finnland", ]
capitals = ["wien", "berlin", "paris", "madrid", "london", "helsinki"]
words = countries + capitals


def plot_words(model, save_plot_filename=None):
    # get word vectors for words to plot
    vecs = model[words]

    # perform PCA
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(vecs)

    # plot words
    pyplot.scatter(pca_vecs[:, 0], pca_vecs[:, 1])

    for country, capital in zip(range(len(countries)), range(len(countries), 2 * len(countries))):
        pyplot.plot(pca_vecs[[country, capital], 0], pca_vecs[[country, capital], 1], linestyle="dashed", color="gray")

    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(pca_vecs[i, 0], pca_vecs[i, 1]))

    # save or show plot
    if save_plot_filename:
        pyplot.savefig(save_plot_filename)
    else:
        pyplot.show()
    pyplot.clf()


def print_most_similar(model, words):
    print("Words most similar to mean of " + str(words))
    for word, score in model.wv.most_similar(positive=words):
        print(word + " " + str(score))


def print_country_for_capitals(model):
    print("Countries guessed for capitals")
    for capital in capitals[1:]:
        country_guess = model.wv.most_similar(positive=[capital, countries[0]], negative=[capitals[0]], topn=1)[0]
        print(capital + ": " + country_guess[0] + " " + str(country_guess[1]))


normal_window_model = Word2Vec.load("normal_window_model")
small_window_model = Word2Vec.load("small_window_model")

plot_words(normal_window_model, "normal_window_countries_capitals.png")
plot_words(small_window_model, "small_window_countries_capitals.png")

print_most_similar(normal_window_model, ["österreich"])
print_most_similar(small_window_model, ["österreich"])

print_country_for_capitals(normal_window_model)
print_country_for_capitals(small_window_model)


