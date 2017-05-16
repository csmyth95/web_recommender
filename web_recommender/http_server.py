#!/usr/bin/env python

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import simplejson
import random
import sys
import time
import os
import argparse
import logging
import ssl

# local imports
import parse_html
import get_history
import cluster_documents
import generate_cert

# Global variables
RECOMMENDED_LINKS = 'recommended_links.json'
PATH_TO_CERT = 'localhost.pem'
PATH_TO_KEY = 'localhost.key'
FILE_PATH = '~/Library/Application Support/Google/Chrome/Default'
doc_clusters = None
doc_cluster_terms = None
# NOTE: Need to use same vectorizer for both fitting and predicting data with Kmeans
vectorizer = None
lsa = None

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(message)s')


class HandleHTTP(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        """Handle GET request to return the recommended links from recommended_links.json"""
        self._set_headers()
        logging.info("INFO: GET received, sending recommended_links.json")
        f = open(RECOMMENDED_LINKS, "r")
        self.wfile.write(f.read())
        # Delete links after they are sent to the front end.
        os.remove(RECOMMENDED_LINKS)
        logging.info("JSON file with recommended links removed")

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        """Handle POST request to use client JSON data to recommend links"""
        self._set_headers()
        logging.info("INFO: POST request received...")
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        # NOTE: data is of type list
        data = simplejson.loads(self.data_string)
        data_documents = parse_html.parse_html(data)
        recommended_links = cluster_documents.compare_items_to_cluster(doc_clusters, data_documents, args, vectorizer, lsa)
        json_response = simplejson.dumps(recommended_links)
        with open(RECOMMENDED_LINKS, "w") as outfile:
            simplejson.dump(json_response, outfile)
        logging.info("INFO: Links to recommend: %s" % json_response)
        self.send_response(200)
        self.end_headers()


def run(server_class=HTTPServer, handler_class=HandleHTTP, port=4443):
    hostname = '127.0.0.1'
    server_address = (hostname, port)
    httpd = server_class(server_address, handler_class)
    # TODO: Get https working with cert, no response atm when ssl wrapper is set
    # NOTE:Try and switch to SimpleHTTPRequestHandler
    #httpd.socket = ssl.wrap_socket (httpd.socket, certfile=PATH_TO_CERT, keyfile=PATH_TO_KEY, server_side=True)
    print()
    print 'Starting httpd...'
    try:
        # NOTE: 2nd httpd for reference
        #httpd = BaseHTTPServer.HTTPServer(('localhost', 4443), SimpleHTTPServer.SimpleHTTPRequestHandler)
        #httpd.socket = ssl.wrap_socket (httpd.socket, certfile='path/to/localhost.pem', server_side=True)
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print time.asctime(time.localtime()), "Server Stops - %s:%s" % (hostname, port)


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file-path',
		'-f',
		default=FILE_PATH,
		help='path to History sqlite file of Chrome.'
    )
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
		'--chrome-url',
		default='https://www.google.ie/',
		help='URL to open after script is run'
	)
    parser.add_argument(
		'--url-limit',
		default=1000,
        type=int,
		help='Set limit for the amount of URLs to parse. Default=%(default)s'
	)
    parser.add_argument(
        '--true-k',
        default=5,
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
        default=False,
        help="Use ordinary k-means algorithm (in batch mode)."
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
        help="Maximum number of features (dimensions) to extract from text."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="logging.info progress reports inside k-means algorithm."
    )
    args = parser.parse_args()
    print(args)
    print("\n")
    # TODO: <AFTER_THOUGHT> add some parameter to switch between different functionality of the script
    generate_cert.create_cert()
    get_history.copy_chrome_history(args.file_path, args.current_dir)
    t_now = time.time()
    urls = parse_html.get_urls(args.current_dir, args.url_limit)
    text_docs = parse_html.parse_html(urls)
    t_after_parse = (time.time() - t_now)/60.0
    logging.info("Parsing took: %0.3f" % t_after_parse)
    doc_clusters, doc_cluster_terms, vectorizer, lsa = cluster_documents.cluster_docs(text_docs, args)
    t_after_clustering = (time.time() - t_after_parse)/60.0
    logging.info("Clustering took: %0.3f" % t_after_clustering)
    logging.info("History collected, parsed and ready for recommendations.")
    get_history.open_chrome(args.chrome_path, args.chrome_url)
    run()
