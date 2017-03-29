#!/usr/bin/env python

"""
Script to cluster HTML text

TODO:
1. Get HTML in dictionary (format: 'url' => '<text>')
2. Cluster data
3. Store cluster data

NOTE: find way to import functions from parse_html.py
"""
from __future__ import print_function
import argparse
import logging
import os
import urllib
import bs4
import textmining
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

import optparse
import sys
from time import time
import numpy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def cluster_docs(text_docs, args, labels=None):
    """Cluster HTML documents using a clustering ML algorithm from sci-kit learn"""

    logging.info("INFO: Extracting features from the dataset using a sparse vectorizer")
    t0 = time()
    # if args.use_hashing:
    #     if args.use_idf:
    #         # Perform an IDF normalization on the output of HashingVectorizer
    #         hasher = HashingVectorizer(
    #             n_features=args.n_features, stop_words='english', non_negative=True,
    #             norm=None, binary=False
    #         )
    #         vectorizer = make_pipeline(hasher, TfidfTransformer())
    #     else:
    #         vectorizer = HashingVectorizer(
    #             n_features=args.n_features, stop_words='english',
    #             non_negative=False, norm='l2', binary=False
    #         )
    # else:
    vectorizer = TfidfVectorizer(
        max_features=args.n_features, stop_words='english', use_idf=args.use_idf
    )
    train_vectorizer = vectorizer.fit(text_docs.values())
    vectorized = train_vectorizer.transform(text_docs.values())

    logging.info("done in %fs" % (time() - t0))
    logging.info("n_samples: %d, n_features: %d" % vectorized.shape)
    logging.info("------------------")

    if args.n_components:
        # TODO: Create function for this so can swap between LDA and LSA
        logging.info("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(args.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        vectorized = lsa.fit_transform(vectorized)

        logging.info("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        logging.info("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

        logging.info("---------------------")

    # NOTE: kmeans++: Selects initial clusters in a way that speeds up convergence
    if args.minibatch:
        km = MiniBatchKMeans(n_clusters=args.true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=args.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=args.verbose)

    logging.info("Clustering sparse data with %s" % km)
    t0 = time()
    # Compute KMeans Clustering
    document_clusters = km.fit(vectorized)

    logging.info("done in %0.3fs" % (time() - t0))
    logging.info("---------------")

    # Explanation: have labels and want to see if the clustering algorithm happened to cluster the data according to your labels
    # NOTE: These can't be used as my documents aren't already labelled
    if labels is not None:
        logging.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, document_clusters.labels_))
        logging.info("Completeness: %0.3f" % metrics.completeness_score(labels, document_clusters.labels_))
        logging.info("V-measure: %0.3f" % metrics.v_measure_score(labels, document_clusters.labels_))
        logging.info("Adjusted Rand-Index: %.3f"
                     % metrics.adjusted_rand_score(labels, document_clusters.labels_))
        # NOTE: Silhouette score: closer to 1 the better, mean of silhouette Coefficient for all observations, large dataset == long time
        logging.info("Silhouette Coefficient: %0.3f"
                     % metrics.silhouette_score(vectorized, document_clusters.labels_, sample_size=1000))
        logging.info("---------------")

    if not args.use_hashing:
        logging.info("Top terms per cluster:")

        if args.n_components:
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
    return document_clusters, terms, train_vectorizer


def compare_items_to_cluster(document_clusters, client_data, args, vectorized):
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
    logging.info("INFO: Extracting features from the client dataset using a sparse vectorizer")
    # fit_transform: Transform a sequence of documents to a document-term matrix.
    client_array = vectorized.transform(client_data.values())
    logging.info("Client Array: ")
    logging.info(client_array)

    # Returns index of the cluster each sample belongs to.
    # TODO: If this doesn't work, change to use km instead of document_clusters
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

    # Find biggest cluster
    # TODO: find 2nd biggest cluster
    logging.info("INFO: Labels and the count")
    biggest_cluster = labels[0]
    cluster_size = unique_labels[biggest_cluster]
    for label in unique_labels:
        if cluster_size < unique_labels[label]:
            biggest_cluster = label
            cluster_size = unique_labels[label]
        # print(str(label) +" - "+str(unique_labels[label]))
    print("INFO: Largest Cluster: "+str(biggest_cluster))

    # TODO: Create links list to recommend
    recommended_links = []
    urls_from_client = client_data.keys()
    counter = 0
    for label in client_labels:
        if label == biggest_cluster:
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
        dest="n_components",
        type=int,
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
    doc_clusters, doc_cluster_terms = cluster_docs(text_docs, args, args.true_k)
    # TODO: new_data should be links from client
    new_data = []
    #compare_items_to_cluster(doc_clusters, new_data, args)
    logging.info("----------------------------")
    logging.info("COMPLETE")


if __name__ == '__main__':
    main()
