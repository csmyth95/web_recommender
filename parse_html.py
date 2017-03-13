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
import urllib
import bs4
import textmining
#import textract

logging.basicConfig(level=logging.INFO)


def get_urls(current_dir):
	"""Read file of urls into a list"""
	filename = os.path.join(current_dir, "history.txt")
	urls = []
	counter = 0
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip()
			if counter > 100:
				break
			# Exclude URLs that aren't http protocols
			if line.startswith('chrome') or line.startswith('file') or line.startswith('data'):
				continue
			# Exclude PDFs until a way to parse them can be found
			if line.endswith('.pdf'):
				continue
			urls.append(line)
			counter += 1
	if urls is None:
		logging.warning('WARNING: urls list is empty')
	return urls


def parse_html(urls):
	"""
	urls: list of urls from user's local history
	"""
	urls_with_text = dict()
	logging.info('Parsing HTML files into text.\n')
	for url in urls:
		logging.info(url)
		# TODO: Create try except for timeouts
		html = urllib.urlopen(url).read()
		"""
		#text = ''
		# TODO: Implement way to parse PDF files
		#if url.endswith('pdf'):
			# TODO
			#parsed_html = StringIO(html)
		#	logging.info(type(html))
		#	text = textract.process(html)
			#text = slate.PDF(html)
		#else:"""
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
	return urls_with_text


def create_document_term_matrix(url_dict):
	"""Create document term matrix with url as document and then the contained words
	NOTES:
	- Should I strip out insignificant words e.g and, to, if, etc...
	"""
    # Initialize class to create term-document matrix
	tdm = textmining.TermDocumentMatrix()
    # Add the documents
	for key in url_dict:
		tdm.add_doc(url_dict[key])
    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
	tdm.write_csv('matrix.csv', cutoff=1)
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
	for row in tdm.rows(cutoff=1):
		logging.info(row)

	# TODO: Implement get most common words
	"""
	# Print ten most common words in the dictionary
    freq_word = [(counts[0][0], word) for (word, counts) in \
      textmining.dictionary.items()]
    freq_word.sort(reverse=True)
    print '\ndictionary_example 1\n'
    for freq, word in freq_word[:10]:
        print word, freq

    # The same word can be used in many different contexts in the English
    # language. The dictionary in the textmining module contains the
    # relative frequencies of each of these parts of speech for a given
    # word. An explanation of the part-of-speech codes is in
    # doc/poscodes.html. Here are the part-of-speech frequencies for the
    # word 'open'.
    print '\ndictionary_example 2\n'
    print textmining.dictionary['open']

	"""


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
	args = parser.parse_args()

	urls = get_urls(args.current_dir)
	text_docs = parse_html(urls)
	logging.info('---------------------------')
	#logging.info(text_docs)
	logging.info(text_docs.keys())
	create_document_term_matrix(text_docs)


if __name__ == '__main__':
	main()
