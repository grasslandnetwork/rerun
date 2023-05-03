import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


class JSONHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') # Add this line to enable CORS
        self.end_headers()
        with open('poses.json', 'r') as f:
            data = json.load(f)
            self.wfile.write(json.dumps(data).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=JSONHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()


if __name__ == '__main__':
    while True:
        run(port=8000)
        time.sleep(1)
