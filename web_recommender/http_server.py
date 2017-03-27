#!/usr/bin/env python

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import simplejson
import random
import sys
from time import time
import parse_html

# Global variables
FILE_PATH='~/Library/Application Support/Google/Chrome/Default'


class HandleHTTP(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        """TODO: Figure out if I need this GET function"""
        f = open("index.html", "r")
        self.wfile.write(f.read())

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        """TODO: Handle JSON data, not stream data"""
        self._set_headers()
        print "in post method"
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        # TODO: parse JSON, create matrix and ML algorithm => send recommendedlinks back
        # Send links back in the response
        # TODO: Find someway to do all the Chrome History before trying to recommend links
        # For now, do everything here
        # NOTE: data is of type list
        data = simplejson.loads(self.data_string)
        with open("user_links.json", "w") as outfile:
            simplejson.dump(data, outfile)
        print "{}".format(data)
        data_documents = parse_html.parse_html(data)
        print data_documents
        # What to do with the data_documents???

        # TODO: Return links as JSON object (list of urls)
        self.send_response(200)
        self.end_headers()
        return


# def getRecommendedLinks(links):
#     """Utilises clusters to recommend links from a list of links"""


def run(server_class=HTTPServer, handler_class=HandleHTTP, port=9000):
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(port=int(sys.argv[1]))
    else:
        # TODO: Start collecting history from here
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
    		default=200,
    		help='Set limit for the amount of URLs to parse. Default=%(default)s'
    	)
        parser.add_argument(
            'true-k',
            default=10,
            help='Number of clusers to create from the user\'s history'
        )
        parser.add_argument(
                "--lsa",
                dest="n_components", type="int",
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
    	# TODO: add some parameter to switch between different functionality of the script
    	get_history.copy_chrome_history(args.file_path, args.current_dir)
        urls = parse_html.get_urls(args.current_dir)
        text_docs = parse_html.parse_html(urls, arguments.url_limit)
        doc_clusters, doc_cluster_terms = cluster_docs(text_docs, opts, args.true_k)
        logging.info("INFO: History collected, parsed and ready for recommendations.")
        get_history.open_chrome(args.chrome_path, args.chrome_url)
        run()
