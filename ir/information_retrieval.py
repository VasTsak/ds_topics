import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_doc(text: str) -> list:

    # clean text
    text = text.replace("\n", " ")
    text = text.strip()
    text = re.sub(r'[^\w]', ' ', text)

    # normalisation, case-folding
    text = text.lower()

    # tokenise
    word_tokens = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_word_tokens = [
        w for w in word_tokens if w not in stop_words]

    # retrieve stem from words
    ps = PorterStemmer()
    stemming = []

    for w in filtered_word_tokens:
        stem = ps.stem(w)
        stemming.append(stem)

    # retrieve lemma from words
    wordnet_lemmatizer = WordNetLemmatizer()

    lemmatization = []
    for w in stemming:
        lemma = wordnet_lemmatizer.lemmatize(w)
        lemmatization.append(lemma)

    return lemmatization

def create_inverted_index(
        documents: dict,
        inverted_index: dict) -> dict:

    for id, doc in documents.items():

        doc = preprocess_doc(doc)

        for w in doc:
            if w in inverted_index:
                if id in inverted_index[w]:
                    inverted_index[w][id] += 1
                else:
                    inverted_index[w][id] = 1
            else:
                inverted_index[w] = {id: 1}

    return inverted_index


def create_sorted_inverted_index(documents: dict) -> dict:

    inverted_index = create_inverted_index(documents, dict())
    sort_inverted_index = dict()
    for i in inverted_index.keys():
        sort_inverted_index[i] = sorted(
            inverted_index[i].items(),
            key=lambda x: x[0], reverse=True)

    return sort_inverted_index


def intersect_posting(ii1: list, ii2: list):

    a1 = [x[0] for x in ii1]
    a2 = [x[0] for x in ii2]
    answers = []
    i = 0
    j = 0
    while i < len(a1) and j < len(a2):
        if a1[i] == a2[j]:
            answers.append((a1[i], sum([ii1[i][1], ii2[j][1]])))
            i += 1
            j += 1
        elif a1[i] < a2[j]:
            i += 1
        else:
            j += 1
    return answers


def sort_frequency(inverted_index: dict, query_list: list):

    term_freq = dict()
    for q in query_list:
        doc_list = inverted_index.get(q, (None, 0))
        term_freq[q] = sum([x[1] for x in doc_list])
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1])
    sorted_terms = [x[0] for x in sorted_terms]
    return sorted_terms


def rest(terms: list):

    try:
        terms = terms[1:]
    except BaseException:
        terms = None
    return terms


def intersect(query: str, sort_inverted_index: dict) -> list:

    query_list = preprocess_doc(query)
    terms = sort_frequency(sort_inverted_index, query_list)
    first_term = terms[0]
    result = sort_inverted_index[first_term]
    # rest of terms
    terms = rest(terms)
    while len(terms) > 0 and len(terms) > 0:
        first_term = terms[0]
        terms = rest(terms)
        result = intersect_posting(result,
                                   sort_inverted_index[first_term])
    return result


def construct_eval_df(
        query_list: list,
        documents: dict,
        labels: tuple) -> pd.DataFrame:

    inverted_index = create_sorted_inverted_index(documents)
    results = pd.DataFrame(
        [],
        columns=[
            "Documents",
            "Frequency",
            "Query",
            "rank"])

    for q in query_list:
        query = preprocess_doc(q)
        temp_results = intersect(query, inverted_index)
        temp_results = pd.DataFrame(
            temp_results, columns=[
                "Documents", "Frequency"])
        temp_results["Query"] = q
        temp_results["rank"] = temp_results["Frequency"].rank(
            method='dense', ascending=False)
        results = pd.concat((results, temp_results), axis=0)

    labels = pd.DataFrame(labels, columns=["Query", "Documents"])
    labels["IsRelevant"] = 1

    results = results.merge(labels, how="left")
    results["IsRelevant"].fillna(0, inplace=True)
    return results


def mean_reciprocal_rank(df: pd.DataFrame) -> float:

    # we have only one or no document matching the information need
    df["reciprocal_rank"] = df["IsRelevant"] / df["rank"]

    mean_rr = df.groupby(["Query"])["reciprocal_rank"].max().mean()

    return mean_rr


def mean_average_precision(df) -> float:

    for q in df["Query"].unique():
        ranks = sorted(df[df["Query"] == q]["rank"].unique())
        query_precision = list()
        for r in ranks:
            precision = df[(df["Query"] == q) & (
                df["rank"] >= r)]["IsRelevant"].max() / r
            query_precision.append(precision)
        ap = sum(query_precision) / len(query_precision)
        query_precision.append(ap)
    map = sum(query_precision) / len(query_precision)
    return map


documents = {
    "Doc 1": "This is a document about information retrieval.",
    "Doc 2": "This is a document about medium topics related to data.",
    "Doc 3": "This is an article related to mediocre tables.",
    "Doc 4": "Completely irrelevant document.",
    "Doc 5": "Pythagoras of Samos[a] (Ancient Greek: Πυθαγόρας ὁ Σάμιος, romanized: Pythagóras ho Sámios, lit. 'Pythagoras the Samian', or simply Πυθαγόρας; Πυθαγόρης in Ionian Greek; c. 570 – c. 495 BC)[b] was an ancient Ionian Greek philosopher and the eponymous founder of Pythagoreanism. His political and religious teachings were well known in Magna Graecia and influenced the philosophies of Plato, Aristotle, and, through them, the West in general. Knowledge of his life is clouded by legend, but he appears to have been the son of Mnesarchus, a gem-engraver on the island of Samos. Modern scholars disagree regarding Pythagoras's education and influences, but they do agree that, around 530 BC, he travelled to Croton in southern Italy, where he founded a school in which initiates were sworn to secrecy and lived a communal, ascetic lifestyle. This lifestyle entailed a number of dietary prohibitions, traditionally said to have included vegetarianism, although modern scholars doubt that he ever advocated complete vegetarianism.",
    "Doc 6": "Leonardo di ser Piero da Vinci[b] (15 April 1452 – 2 May 1519) was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect.[3] While his fame initially rested on his achievements as a painter, he also became known for his notebooks, in which he made drawings and notes on a variety of subjects, including anatomy, astronomy, botany, cartography, painting, and paleontology. Leonardo is widely regarded to have been a genius who epitomized the Renaissance humanist ideal,[4] and his collective works comprise a contribution to later generations of artists matched only by that of his younger contemporary, Michelangelo.",
    "Doc 7": "Another document about information retrieval. Information retrieval is very informative. It offers you information.",
    "Doc 8": "Another document about information.",
    "Doc 9": "Arithmetic is the branch of mathematics that deals with the study of numbers using various operations on them, which is taught in elementary school. Algebra is taught in high school which is also branch of mathematics.",
    "Doc 10": "Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures, shapes and the spaces in which they are contained, and quantities and their changes.",
    "Doc 11": "Geometry is branch of mathematics. Calculus is branch of mathematics. Discrete mathematics is branch of mathematics."
}
