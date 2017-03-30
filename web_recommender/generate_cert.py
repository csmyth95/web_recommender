#!/usr/bin/env python

import sys
import logging
import socket
import os

from OpenSSL import crypto, SSL
from time import gmtime, mktime

# global variables
CERT_FILE = "localhost.pem"
KEY_FILE = "localhost.key"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(message)s')


def create_cert():
    """Create self signed certificate for HTTP server"""
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        try:
            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 1024)

            # create a self-signed cert
            cert = crypto.X509()
            cert.get_subject().C = "IE"
            cert.get_subject().ST = "Galway"
            cert.get_subject().L = "Galway"
            cert.get_subject().O = "NUIG"
            cert.get_subject().OU = "Computer Science & IT"
            cert.get_subject().CN = socket.gethostname()
            cert.set_serial_number(1000)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(10*365*24*60*60)
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(k)
            cert.sign(k, 'sha1')

            open(CERT_FILE, "wt").write(
                crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            open(KEY_FILE, "wt").write(
                crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
        except crypto.Error as err:
            logging.warning("OpenSSL error occurred: %s" % err)
            return
        except Exception as e:
            logging.warning("Error occurred when creating self signed cert: %s" % e)
            return
    else:
        logging.info("Signed cert already exists.")


def main():
    logging.info("enerating self signed cert for HTTPS websites.")
    create_cert()


if __name__ == '__main__':
    main()
