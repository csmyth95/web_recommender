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


def open_chrome(chrome_path, url=None):
	logging.info('Opening Chrome Browser')
	if sys.platform.startswith('darwin'):
		chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
	if sys.platform.startswith('win32'):
		chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
	if sys.platform.startswith('linux'):
		chrome_path = '/usr/bin/google-chrome %s'
	webbrowser.get(chrome_path).open(url)


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
	open_chrome(args.chrome_path)


if __name__ == '__main__':
	main()
