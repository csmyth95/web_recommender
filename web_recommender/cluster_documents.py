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
#from sklearn.datasets import fetch_20newsgroups
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


def cluster_docs(text_docs, opts):
    """Cluster HTML documents using a clustering ML algorithm from sci-kit learn"""
    labels = text_docs.keys()
    true_k = numpy.unique(labels).shape[0] # Get total number of urls
    logging.info("Extracting features from the dataset using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(
                n_features=opts.n_features, stop_words='english', non_negative=True,
                norm=None, binary=False
            )
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(
                n_features=opts.n_features, stop_words='english',
                non_negative=False, norm='l2', binary=False
            )
    else:
        vectorizer = TfidfVectorizer(
            max_df=0.5, max_features=opts.n_features, min_df=2,
            stop_words='english', use_idf=opts.use_idf
        )
    X = vectorizer.fit_transform(text_docs.values())

    logging.info("done in %fs" % (time() - t0))
    logging.info("n_samples: %d, n_features: %d" % X.shape)
    logging.info("------------------")

    if opts.n_components:
        logging.info("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        logging.info("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        logging.info("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

        logging.info("---------------------")

    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose)

    logging.info("Clustering sparse data with %s" % km)
    t0 = time()
    # Compute KMeans Clustering
    document_clusters = km.fit(X)
    logging.info("done in %0.3fs" % (time() - t0))
    logging.info("---------------")

    # Explanation: have labels and want to see if the clustering algorithm happened to cluster the data according to your labels
    logging.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, document_clusters.labels_))
    logging.info("Completeness: %0.3f" % metrics.completeness_score(labels, document_clusters.labels_))
    logging.info("V-measure: %0.3f" % metrics.v_measure_score(labels, document_clusters.labels_))
    logging.info("Adjusted Rand-Index: %.3f"
                 % metrics.adjusted_rand_score(labels, document_clusters.labels_))
    # NOTE: Silhouette score: closer to 1 the better, mean of silhouette Coefficient for all observations, large dataset == long time
    logging.info("Silhouette Coefficient: %0.3f"
                 % metrics.silhouette_score(X, document_clusters.labels_, sample_size=1000))

    logging.info("---------------")

    if not opts.use_hashing:
        logging.info("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(document_clusters.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print(" Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print
    return document_clusters, terms, true_k


def compare_items_to_cluster(document_clusters, Y, true_k):
    """Checks if any of the top clusters are suitable to enter

    Returns list of suitable urls to use as recommendations.

    Requires Kmeans object
    Params:
    - opts:
    - X: training instances to cluster (needs vectorizarion and LSA first [Outout from cluster_docs?])
    - Y: new training samples (test data or data from user)

    Website: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    # NOTE: kmeans++: Selects initial clusters in a way that speeds up convergence
    #if opts.minibatch:
    #    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
    #                          init_size=1000, batch_size=1000, verbose=opts.verbose)
    #else:
    #    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
    #                verbose=opts.verbose)
    # Compute clusters
    # TODO: Use this as an input, get from cluster_docs()
    #kmeans = km.fit(X)

    # TODO: Use km.predict(Y) Predict the closest cluster each sample in Y belongs to.
    y_labels = document_clusters.predict(Y)

    # TODO: Create links list to recommend
    recommended_links = []

    # TODO: Find largest clusters in kmeans OR check how far away new data is from clusters
    # i.e if data is past a threshold, it is not a good recommendation

    # Cluster centres for each model
    k_means_var = [km.fit(X) for k in true_k]
    centroids = [kmeans.cluster_centers_ for X in k_means_var]
    # Calculate euclidean distance from each point to each cluster centre
    # pairwise(): compute pairwise distances between two points
    k_euclid = [DistanceMetric.get_metric('euclidean').pairwise(X, cent) for cent in centroids]
    dist = [numpy.min(ke, axis=1) for ke in k_euclid]

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
    arguments = parser.parse_args()

    op = optparse.OptionParser()
    op.add_option(
            "--lsa",
            dest="n_components", type="int",
            help="Preprocess documents with latent semantic analysis."
    )
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="logging.info progress reports inside k-means algorithm.")
    (opts, args) = op.parse_args()
    if len(args) > 0:
        logging.info('EXITING')
        op.error('This script takes no arguments.')
        sys.exit(1)

    urls = parse_html.get_urls(arguments.current_dir)
    text_docs = parse_html.parse_html(urls, arguments.url_limit)
    logging.info('---------------------------')
    logging.info(text_docs.keys())
    doc_clusters, doc_cluster_terms, true_k = cluster_docs(text_docs, opts)
    # TODO: new_data should be links from client
    new_data = []
    compare_items_to_cluster(doc_clusters, new_data, true_k)


if __name__ == '__main__':
    main()
