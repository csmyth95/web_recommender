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

logging.basicConfig(level=logging.INFO,
					format='%(levelname)s %(message)s')


def get_urls(current_dir, url_limit):
	"""Read file of urls into a list"""
	filename = os.path.join(current_dir, "history.txt")
	urls = []
	counter = 0
	logging.info("URL Limit: %s" % url_limit)
	with open(filename, 'r') as f:
		for line in f:
			if counter > url_limit:
				logging.info("URL limit reached")
				break
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
		logging.warning('urls list is empty')
	return urls


def parse_html(urls):
	""" Parse text documents from websites

	urls: list of urls from user's local history
	"""
	urls_with_text = dict()
	logging.info('Parsing HTML files into text.\n')
	for url in urls:
		logging.info(url)
		try:
			response = urllib2.urlopen(url, timeout=15)
			# Check if url is requesting authentication
			auth = response.info().getheader('WWW-Authenticate')
			if auth and auth.lower().startswith('basic'):
				logging.warning("Requesting {} requires basic authentication".format(url))
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
			# drop double and single inverted commas
			text = text.replace('""','').replace("''", "")

			urls_with_text[url] = text
		except ssl.CertificateError as e:
			logging.warning(str(e))
			continue
		except Exception as e:
			logging.warning(str(e))
			pass
	return urls_with_text


def main():
	cwd = os.getcwd()

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--current-dir',
		default=cwd,
		help='The current working directory where this script is being run.'
	)
	parser.add_argument(
		'--url-limit',
		default=10,
		type=int,
		help='Set limit for the amount of URLs to parse. Default=%(default)s'
	)
	args = parser.parse_args()

	urls = get_urls(args.current_dir, args.url_limit)
	text_docs = parse_html(urls)
	logging.info('---------------------------')
	#logging.info(text_docs)
	logging.info(text_docs.keys())


if __name__ == '__main__':
	main()
