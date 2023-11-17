import pandas as pd
from os import system, getcwd
from string import punctuation
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from yake import KeywordExtractor
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random
import os
import PIL

STOPWORDS = set(stopwords.words("english"))
LANGUAGE = "en"
MAX_NGRAM_SIZE = 1  # Size of keywords, more than 1 to get phrases.
# Rate to avoid like-terms when picking out keywords. Should be less than 1.
DEDUPLICATION_THRESHSOLD = 0.9
NUM_OF_KEYWORDS = 5  # Number of keywords to retrieve per corpus.

chunk_master_dict = {}
chunk_master_list = []

image_object_array = []

top_nodes = []
bottom_nodes = []
icons = {}


class ImageObject:
    def __init__(self, responses=[], keywords=[], id=-1):
        self._responses: list = responses
        self._keywords: list = keywords
        self._id: int = id

    def get_image_id(self) -> int:
        return self._id

    def get_keywords(self) -> list:
        return self._keywords


def run_semantic_analysis_demo():
    system("cls")

    response_list = []

    annotations = ["/annotations/Image_Annotations_Set_1.csv",
                   "/annotations/Image_Annotations_Set_2.csv",
                   "/annotations/Image_Annotations_Set_3.csv"]

    for file in annotations:
        df = pd.read_csv(getcwd() + file)
        rli = [df[str(col + 1)] for col in range(df.shape[1] - 1)]
        response_list.extend(rli)

    processed_response_sets = []

    idx = 1
    for response_set in response_list:

        for chunk in extract_chunks(response_set):
            chunk_master_dict[chunk] = ''

        chunk_master_list = list(chunk_master_dict.keys())
        chunk_master_list = [
            chunk for chunk in chunk_master_list if " " in chunk]

        extractor = KeywordExtractor(
            lan=LANGUAGE,
            n=MAX_NGRAM_SIZE,
            dedupLim=DEDUPLICATION_THRESHSOLD,
            top=NUM_OF_KEYWORDS,
            features=None)

        keywords = []
        for response in response_set:
            raw_keywords_set = extractor.extract_keywords(response)

            new_keyword_set = []
            for raw_keyword_tup in raw_keywords_set:
                kw = raw_keyword_tup[0]

                for chunk in chunk_master_list:
                    if kw in chunk:
                        kw = chunk

                new_keyword_set.append(kw.lower())
                new_keyword_set = [*set(new_keyword_set)]
            keywords.append(new_keyword_set)
            processed_response = [process_text(
                response) for response in response_set]
            processed_response_sets.append(processed_response)

        image_object_array.append(ImageObject(
            response_set, new_keyword_set, idx))
        idx += 1

    # Initialize graph and load keywords (no repeats) into nodes
    B = nx.Graph()
    bottom_nodes = get_master_kw_list(image_object_array)

    # Read images into memory for displaying in the graph
    load_images()
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}

    # Generate a list of 5 random images to display in the graph
    demo_images = get_random_images()

    # Create edges between image nodes and keyword nodes
    i = 1
    for image_object in image_object_array:
        keywords = image_object.get_keywords()
        if i in demo_images:
            top_nodes.append(i)
            color = random_color()
            B.add_node(i, image=images[str(i)])
            for kw in keywords:
                if kw in bottom_nodes:
                    B.add_edge(i, kw, color=color)
        i += 1

    # Update position for node from each group
    left, right = nx.bipartite.sets(B, top_nodes=top_nodes)
    pos = {}

    i = len(right) * 3.475
    for node in right:
        pos[node] = (2, i)
        i -= 3.75

    i = (len(top_nodes) * 9.25)
    for node in left:
        pos[node] = (1, i)
        i -= 9

    fig, ax = plt.subplots()
    edges = B.edges()
    edge_colors = [B[u][v]['color'] for u, v in edges]

    nx.draw(B, pos=pos, with_labels=True, node_color=(0.8, 0.8, 0.8),
            edge_color=edge_colors, font_size=15, width=4)

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform

    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.12
    icon_center = icon_size / 2.0

    # Position photos over image nodes
    is_int = True
    for n in B.nodes:
        try:
            int(n)
        except ValueError:
            is_int = False
        if is_int:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes(
                [xa - icon_center, ya - icon_center, icon_size, icon_size])
            a.imshow(B.nodes[n]["image"])
            a.axis("off")
        is_int = True

    # display final graph
    plt.show()
# ------------------------------------------------------------- #


def extract_chunks(response_set: str) -> dict:
    chunked_token = ""
    chunk_dict = {}

    for sentence in response_set:
        for sent in sent_tokenize(sentence):
            for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    chunked_token = ' '.join(c[0] for c in chunk)

        chunk_dict[chunked_token] = ""

    return chunk_dict


def process_text(text: str) -> list:
    tagged = pos_tag(word_tokenize(text))
    lemmatizer = WordNetLemmatizer()

    lemmatized_words = []

    for tup in tagged:
        if pos_sorter(tup[1]) != None:
            result = lemmatizer.lemmatize(tup[0], pos_sorter(tup[1]))
            lemmatized_words.append(result)

    processed_tokens = [word for word in lemmatized_words
                        if word not in STOPWORDS and not
                        word.isdigit() and word not in punctuation]

    return processed_tokens


def pos_sorter(word_tag):
    if word_tag.startswith('J'):
        return wordnet.ADJ
    elif word_tag.startswith('V'):
        return wordnet.VERB
    elif word_tag.startswith('N'):
        return wordnet.NOUN
    elif word_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# create a master list of the keywords without duplicates for graphing
def get_master_kw_list(objectArray: list):
    master_kw_list = []
    for image_object in image_object_array:
        for kw in image_object.get_keywords():
            if kw not in master_kw_list:
                master_kw_list.append(kw)
    return master_kw_list


def random_color():
    rgb = []
    for i in range(3):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgb = [r, g, b]
    return rgb


def get_random_images():
    random_images = []
    i = 0
    while i < 4:
        num = random.randint(1, 70)
        if num not in random_images:
            random_images.append(num)
            i += 1
    return random_images


def load_images():
    directory = 'images'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            pathname, extension = os.path.splitext(f)
            fname = pathname.split('/')
            icons[fname[-1]] = f


if __name__ == "__main__":
    run_semantic_analysis_demo()
