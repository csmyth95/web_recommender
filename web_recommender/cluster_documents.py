#!/usr/bin/env python

"""
Script to cluster HTML text documents

1. Get HTML in dictionary (format: 'url' => '<text>')
2. Use vectorization and dimensionality reduction technique e.g LSA
3. Cluster data
4. Create function for predicting new data and return recommendations

NOTE: find way to import functions from parse_html.py
"""
from __future__ import print_function
import argparse
import logging
import os
import urllib
import bs4
import parse_html
# import textract

# Clustering imports
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import DistanceMetric

import sys
from time import time
import numpy

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(message)s')


def cluster_docs(text_docs, args, labels=None):
    """Cluster HTML documents using a clustering ML algorithm from sci-kit learn"""

    logging.info("Extracting features from the dataset using a sparse vectorizer")
    t0 = time()
    # Choose vectorizer to use
    if args.use_hashing:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(
            n_features=args.n_features, stop_words='english', non_negative=True,
            norm=None, binary=False
        )
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = TfidfVectorizer(
            max_features=args.n_features, stop_words='english', use_idf=args.use_idf
        )
    logging.info("Using vectorization function: %s" % vectorizer)

    train_vectorizer = vectorizer.fit(text_docs.values())
    vectorized = train_vectorizer.transform(text_docs.values())

    vectorization_time = (time() - t0)/60.0
    logging.info("done in %0.3f" % vectorization_time)
    logging.info("n_samples: %d, n_features: %d" % vectorized.shape)
    logging.info("------------------")

    lsa = None
    if args.lsa:
        logging.info("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        # 100 features is recommended for LSA
        svd = TruncatedSVD(n_components=100)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        vectorized = lsa.fit_transform(vectorized)
        lsa_time = (time() - t0)/60.0
        logging.info("done in %0.3f" % lsa_time)

        explained_variance = svd.explained_variance_ratio_.sum()
        logging.info("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))
        logging.info("After dimensionality reduction:")
        logging.info("n_samples: %d, n_features: %d" % vectorized.shape)
        logging.info("---------------------")

    # NOTE: kmeans++: Selects initial clusters in a way that speeds up convergence
    # Choose between computational efficiency or cluster quality
    if args.minibatch:
        km = MiniBatchKMeans(
            n_clusters=args.true_k, init='k-means++', n_init=1,
            init_size=1000, batch_size=1000, verbose=args.verbose
        )
    else:
        km = KMeans(
            n_clusters=args.true_k, init='k-means++', max_iter=100,
            n_init=1, verbose=args.verbose
        )

    logging.info("Clustering sparse matrix with: %s" % km)
    t0 = time()
    document_clusters = km.fit(vectorized)
    clustering_time = (time() - t0)/60.0
    logging.info("Clustering done in %0.3f" % clustering_time)
    logging.info("---------------")

    if not args.use_hashing:
        logging.info("Top terms per cluster:")

        if args.lsa:
            original_space_centroids = svd.inverse_transform(document_clusters.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(args.true_k):
            print(" Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print
    return document_clusters, terms, train_vectorizer, lsa


def compare_items_to_cluster(document_clusters, client_data, args, train_vectorizer, lsa):
    """Checks if any of the top clusters are suitable to enter

    Returns list of suitable urls to use as recommendations.

    Requires Kmeans object
    Params:
    - args:
    - document_clusters: clusters of documents from user's history
    - client_data: new training samples (test data or data from user) [dict with url as key, text as value]
    - true_k: number of clusters
    - vectorized: used to transform the client data into a document term matrix

    Website: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    # Vectorize the client data for predictions
    logging.info("Extracting features from the client dataset using a sparse vectorizer")
    # fit_transform: Transform a sequence of documents to a document-term matrix.
    vectorized_client = train_vectorizer.transform(client_data.values())
    client_array = lsa.transform(vectorized_client)

    # Returns index of the cluster each sample belongs to.
    client_labels = document_clusters.predict(client_array)
    logging.info("Predictions")
    logging.info(client_labels)
    labels = document_clusters.labels_
    unique_labels = dict()
    for label in labels:
        if unique_labels.has_key(label):
            unique_labels[label] += 1
        else:
            unique_labels[label] = 1

    # Find biggest and 2nd biggest clusters
    logging.info("Labels and the count")
    biggest_cluster = labels[0]
    second_largest = biggest_cluster
    cluster_size = unique_labels[biggest_cluster]
    for label in unique_labels:
        if cluster_size < unique_labels[label]:
            second_largest = biggest_cluster
            biggest_cluster = label
            cluster_size = unique_labels[label]
    logging.info("Largest Cluster: "+str(biggest_cluster)+" 2nd Largest: "+str(second_largest))

    # Create links list to recommend
    recommended_links = []
    urls_from_client = client_data.keys()
    counter = 0
    for label in client_labels:
        if label == biggest_cluster or label == second_largest:
            recommended_links.append(urls_from_client[counter])
        counter += 1

    return recommended_links


def main():
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--chrome-path',
        default='open -a /Applications/Google\ Chrome.app %s',
        help='set chrome path for the specific OS. Default=%(default)s'
    )
    parser.add_argument(
        '--current-dir',
        default=cwd,
        help='The current working directory where this script is being run.'
    )
    parser.add_argument(
		'--url-limit',
		default=200,
		help='Set limit for the amount of URLs to parse. Default=%(default)s'
	)
    parser.add_argument(
        '--true-k',
        default=10,
        help='Number of clusers to create from the user\'s history'
    )
    parser.add_argument(
        "--lsa",
        action="store_true",
        default="True",
        help="Preprocess documents with latent semantic analysis."
    )
    parser.add_argument(
        "--no-minibatch",
        action="store_false",
        dest="minibatch",
        default=True,
        help="Use ordinary k-means algorithm (in batch mode)."
    )
    parser.add_argument(
        "--use-idf",
        action="store_false",
        default=True,
        help="Disable Inverse Document Frequency feature weighting."
    )
    parser.add_argument(
        "--use-hashing",
        action="store_true",
        default=False,
        help="Use a hashing feature vectorizer"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=10000,
        help="Maximum number of features (dimensions) to extract from text.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="logging.info progress reports inside k-means algorithm."
    )
    args = parser.parse_args()

    urls = parse_html.get_urls(args.current_dir)
    text_docs = parse_html.parse_html(urls, args.url_limit)
    logging.info('---------------------------')
    logging.info(text_docs.keys())
    doc_clusters, doc_cluster_terms, train_vectorizer, lsa = cluster_docs(text_docs, args, args.true_k)
    logging.info("----------------------------")
    logging.info("COMPLETE")


if __name__ == '__main__':
    main()
