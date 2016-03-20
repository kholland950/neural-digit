from sys import version as python_version
from cgi import parse_header
if python_version.startswith('3'):
    from urllib.parse import parse_qs
    from http.server import BaseHTTPRequestHandler
else:
    from urlparse import parse_qs
    from BaseHTTPServer import BaseHTTPRequestHandler
    from BaseHTTPServer import HTTPServer
import json
import pndigit
import numpy as np

net = pndigit.loadNet("final.net")

class DigitHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        postvars = self.parse_POST()
        pixels = postvars['pixels[]']
        features = []
        features.append(np.array(pixels))
        features = np.array(features)
        val = net.predict_label(features)[0]
        print(val)
        self.send_response(200)
        self.send_header("Content-type","text/plain")
        self.send_header("Access-Control-Allow-Origin","*");
        self.end_headers()
        self.wfile.write(val)

    def parse_POST(self):
        ctype, pdict = parse_header(self.headers.getheader('content-type'))
        if ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(self.rfile.read(length), keep_blank_values=1)
        return postvars

server = HTTPServer(('', 8090), DigitHandler)
server.serve_forever()

