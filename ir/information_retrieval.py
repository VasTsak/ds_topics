import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

import create_log

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# create the logs, comment out if you don't need them 
create_log.main()

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
            key=lambda x: x[1], reverse=True)

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
    result.sort(key = lambda x: x[1], reverse=True)
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


def mean_average_precision(df: pd.DataFrame) -> float:

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

def main():
        
    logs = open("data/app.log")
    line = dict()
    for i, l in enumerate(logs):
        line[i] = l

    query = "error model"
    inverted_index = create_sorted_inverted_index(line)
    result = intersect(query, inverted_index)
    return result

if __name__=="__main__":
    main()