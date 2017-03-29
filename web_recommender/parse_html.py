#!/usr/bin/env python

"""
Script to parse HTML to text

TODO:
1. Get URLs from history text file
2. Read in HTML docs and parse out text
3. Send text with url to cassandra and or solr
"""
import argparse
import logging
import os
import urllib2
import bs4
import ssl
from time import time
# import textract

logging.basicConfig(level=logging.INFO)


def get_urls(current_dir):
	"""Read file of urls into a list"""
	filename = os.path.join(current_dir, "history.txt")
	urls = []
	counter = 0
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip()
			if line.endswith('pdf'):
				# TODO: Implement way to parse PDFs correctly
				continue
			if line.startswith('http'):
				urls.append(line)
				counter += 1
			else:
				# Exclude URLs that don't use the http protocols
				continue
	if urls is None:
		logging.warning('WARNING: urls list is empty')
	return urls


def parse_html(urls, url_limit):
	"""
	urls: list of urls from user's local history

	     TODO:
		 - Check if url is reachable e.g urls from HPE
		 - Filter out URLs that look for authentication e.g blackboard
		 - - What about sites like Facebook?
		 - Implement way to parse PDF files
	     - Create try except for timeouts
	"""
	urls_with_text = dict()
	logging.info('Parsing HTML files into text.\n')
	t0 = time()
	counter = 0
	for url in urls:
		logging.info(url)
		try:
			response = urllib2.urlopen(url, timeout=15)
			# Check if url is requesting authentication
			auth = response.info().getheader('WWW-Authenticate')
			if auth and auth.lower().startswith('basic'):
				logging.warning("WARNING: Requesting {} requires basic authentication".format(url))
				continue

			html = response.read()
			soup = bs4.BeautifulSoup(html, 'lxml')

			# Rip out all script and style elements
			for script in soup(["script", "style"]):
				script.extract()

			# get text
			text = soup.get_text()
			# break into lines and remove leading and trailing space on each
			lines = (line.strip() for line in text.splitlines())
			# break multi-headlines into a line each
			chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
			# drop blank lines
			text = '\n'.join(chunk for chunk in chunks if chunk)
			urls_with_text[url] = text
			counter += 1
			if counter > url_limit:
				break
		except ssl.CertificateError as e:
			logging.warning("WARNING: "+str(e))
			continue
		except Exception as e:
			logging.warning("WARNING: "+str(e))
			pass

	logging.info("INFO: Parsing took %0.3fs" % (time() - t0))
	return urls_with_text


def main():
	cwd = os.getcwd()

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--chrome-path',
		default='open -a /Applications/Google\ Chrome.app %s',
		help='set chrome path for the specific OS. Default=%(default)s'
	)
	parser.add_argument(
		'--keyspace',
		default='recommendersystem',
		help='Apache Cassandra keyspace to use. Default=%(default)s'
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
	args = parser.parse_args()

	urls = get_urls(args.current_dir)
	text_docs = parse_html(urls, args.url_limit)
	logging.info('---------------------------')
	#logging.info(text_docs)
	logging.info(text_docs.keys())
	create_document_term_matrix(text_docs)


if __name__ == '__main__':
	main()
