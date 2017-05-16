# web_recommender
A recommendation system based on web browsing. Uses a simple Python HTTP server to serve requests from
a JavaScript client running in a web browser.

NOTE:
- Only supports MAC OSX.
- Only supports Google Chrome browser.
- Does not work on sites running HTTPS.

# Setup
## Server Side
Firstly, clone this repository:

```
git clone <repo_url>
```

To resolve all dependencies for the repository, execute this command from the terminal, from the top directory of the repository:

```
make
```

When all dependencies have been resolved, start the HTTP server from the command line to begin collecting your chrome history data.
```
./web_recommender/http_server.py
```

When the server script has finished and the clusters have been collected, it can now serve requests from the browser plugin on the client side. 

## Client side
The client side is a UserScript for the Tampermonkey plugin. The following instructions will show how to
install it for your system.

### Install Tampermonkey extension
Install the stable version of Tampermonkey from the tampermonkey homepage:

[TamperMonkey](https://tampermonkey.net/)

### Import the tampermonkey script
In the base directory for this repository is a tampermonkey script to be imported into the TamperMonkey plugin. To import the script, click on the TamperMonkey plugin icon, then click on "Dashboard". From the Dashboard, go to "Utilities" and click the "Import file" button in the file section and select:
```
tampermonkey_script.txt
```
Once imported, the script should show up in the "Installed userscripts" section.

### Run the script
First, ensure the local http server is running, then go to a website where want to get recommended links. Click on the tampermonkey plugin, switch the plugin on and reload the current page to start generating the recommendations.

NOTE: recommendations work best with higher training data e.g >1000 urls from user history.

## Tools used during development

* [sci-kit learn](http://scikit-learn.org/stable/index.html) - Machine Learning package for Python.
* [Tampermonkey](https://tampermonkey.net/) - Platform to develop front end JavaScript client.
* [BaseHTTPServer](https://docs.python.org/2/library/basehttpserver.html) - Package used to create the HTTP server.
