#!/usr/bin/env python

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import simplejson
import random
import sys
from time import time
import parse_html


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


def getRecommendedLinks(links):
    """Utilises clusters to recommend links from a list of links"""


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
        logging.info("INFO: History collected, parsed and ready for recommendations.")
        run()
