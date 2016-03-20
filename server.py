from sys import version as python_version
from cgi import parse_header
if python_version.startswith('3'):
    from urllib.parse import parse_qs
    from http.server import BaseHTTPRequestHandler
else:
    from urlparse import parse_qs
    from BaseHTTPServer import BaseHTTPRequestHandler
    from BaseHTTPServer import HTTPServer
import digit
import json
import pndigit
import numpy as np

net = pndigit.buildNet()

class DigitHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        postvars = self.parse_POST()
        pixels = postvars['pixels[]']
        features = []
        features.append(np.array(pixels))
        features = np.array(features)
        print(net.predict_label(features)[0])


    def parse_POST(self):
        ctype, pdict = parse_header(self.headers.getheader('content-type'))
        if ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(self.rfile.read(length), keep_blank_values=1)
        return postvars

server = HTTPServer(('', 8090), DigitHandler)
server.serve_forever()

