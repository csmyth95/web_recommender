#!/usr/bin/env python

# Todo:
# 1. Copy history file to current directory
# 2. Send history to a local text file
# 3. Open Chrome...

import argparse
import logging
import os
import webbrowser
import sys
import re
import sqlite3
import shutil

# Local imports
import parse_html

logging.basicConfig(level=logging.INFO)

# global variables
FILE_PATH='~/Library/Application Support/Google/Chrome/Default'
TEXT_FILE='history.txt'
TABLE_NAME = "parsed_urls"


def copy_chrome_history(path_to_file, current_dir):
	"""Copy urls from local History sqlite3 db to local text file

	The local History file of Google Chrome is an SQLite database. This function
	reads the contents of the urls table into a file called history.txt in the current
	directory.

	NOTE: For sqlite command to work, all instances of Chrome must be shutdown.
	"""
	history_file = current_dir+'/'+TEXT_FILE
	original_file_path = os.path.expanduser(path_to_file)

	if os.path.exists(original_file_path):
		try:
			history_db = original_file_path+'/History'
			history_dest = current_dir + '/History'
			shutil.copyfile(history_db, history_dest)
			logging.info('History sqlite file copied to: '+original_file_path)
			connection = sqlite3.connect('History')
			logging.info('Database connected to successfully')
			cursor = connection.execute("SELECT url FROM urls ORDER BY last_visit_time DESC")
			with open(history_file, 'w') as f:
				for row in cursor:
					f.write(row[0]+'\n')
			logging.info('History database has been copied to: '+history_file)
			connection.close()
			logging.info('sqlite3 connection closed.')
		except sqlite3.Error as e:
			logging.error('Sqlite3 error: '+str(e))
	else:
		logging.error('ERROR: Directory for the History file does not exist.')
		raise SystemExit(1)


# def send_to_db(urls_with_text):
#  	"""Send URLs and text obtained from Chrome History to sqlite3 database.
#
# 	Parameters:
# 	- urls_with_text: dictionary of the form {url: "text document"}
# 	"""
#
# 	try:
# 		connection = sqlite3.connect('History')
# 		logging.info('Database connected to successfully')
# 		# if table exists, drop it to store only the latest user browsing history
# 		logging.info("Dropping table if it exists already...")
# 		drop_table = "drop table if exists %s" % (TABLE_NAME)
# 		connection.execute(drop_table)
# 		# Create table
# 		logging.info("Creating table to store urls and parsed text from History.")
# 		connection.execute('''CREATE TABLE %s(
# 			ID		    INT PRIMARY KEY     NOT NULL,
# 			URL 	    TEXT    NOT NULL,
# 			DOCUMENT	TEXT     NOT NULL);''' % (TABLE_NAME)
# 		)
#
# 		counter = 0
# 		logging.info("Table created, insert user history data")
# 		# TODO: Fix insertion error
# 		cursor = connection.cursor()
# 		for url in urls_with_text:
# 			statement = ('INSERT INTO %s (ID, URL, DOCUMENT) '
# 			'VALUES (%d, "%s", "%s")') % (TABLE_NAME, counter, url, urls_with_text[url])
# 			logging.info("Statement for insertion: "+statement)
# 			cursor.execute(statement)
# 			counter += 1
# 		logging.info("Values ready to be inserted, commit the transaction")
# 		connection.commit()
# 		connection.close()
# 		logging.info('sqlite3 connection closed.')
# 	except sqlite3.Error as e:
# 		logging.error('Sqlite3 error: '+str(e))
# 		logging.error(sys.exc_info())
# 		sys.exit(1)
#
#
# def retrieve_from_db():
# 	"""Retrieve stored urls and text documents from local SQLite database
#
# 	Returns dictionary of the form: {url: "text document"}
# 	"""
# 	urls_dict = dict()
# 	try:
# 		connection = sqlite3.connect('History')
# 		logging.info('Database connected to successfully')
# 		cursor = connection.execute("SELECT url, document FROM if exists %s ORDER BY id ASC" % TABLE_NAME)
# 		for row in cursor:
# 			urls_dict[row[1]] = row[2]
# 		logging.info('Database entries retrieved from: %s' % TABLE_NAME)
# 		logging.info(urls_dict)
# 		connection.close()
# 		logging.info('sqlite3 connection closed.')
# 	except sqlite3.Error as e:
# 		logging.error('Sqlite3 error: '+str(e))
# 		sys.exit(1)
# 	sys.exit(1)
# 	return urls_dict
#
#
# def open_chrome(chrome_path, url=None):
# 	logging.info('Opening Chrome Browser')
# 	if sys.platform.startswith('darwin'):
# 		chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
# 	if sys.platform.startswith('win32'):
# 		chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
# 	if sys.platform.startswith('linux'):
# 		chrome_path = '/usr/bin/google-chrome %s'
# 	webbrowser.get(chrome_path).open(url)


def clean_up():
	"""Clean up current directory
	"""
	# TODO: remove History file, history.txt and other files which aren't needed after data is collected


def main():
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
	args = parser.parse_args()
	copy_chrome_history(args.file_path, args.current_dir)
	urls = parse_html.get_urls(args.current_dir)
	text_docs = parse_html.parse_html(urls, args.url_limit)
	send_to_db(text_docs)
	urls_with_text = retrieve_from_db()


if __name__ == '__main__':
	main()
