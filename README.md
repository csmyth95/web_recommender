# web_recommender
A recommendation system based on web browsing. Uses a simple Python HTTP server to serve requests from
a JavaScript client running in a web browser.

NOTE:
- Only supports MAC OSX
- Google Chrome is the only supported browser.

## Setup
Firstly, clone this repository using

```
git clone
```

To resolve all dependencies for the package, execute this command in your terminal:

```
make
```

When all dependencies have been resolved, start the HTTP server from the command line to begin collecting your chrome history data.
```
./web_recommender/http_server.py
```


When the server is finished and the clusters have been collected, it can now server requests from the browser plugin. 

## Client side
###TODO
The client side is a UserScript for the Tampermonkey plugin. The following instructions will show how to
install it for your system.

### Install Tampermonkey extension
TODO

## Built With

* [sci-kit learn](http://scikit-learn.org/stable/index.html) - Machine Learning package for Python.
* [Tampermonkey](https://tampermonkey.net/) - Platform to develop front end JavaScript client.
* [BaseHTTPServer](https://docs.python.org/2/library/basehttpserver.html) - Package used to create the HTTP server.
