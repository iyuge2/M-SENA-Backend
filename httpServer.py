import os
import multiprocessing
from http.server import HTTPServer,CGIHTTPRequestHandler

from constants import *

def http_server(path, port):
    try:
        os.chdir(path)
        server_address = ("",port) # 设置服务器地址
        server_obj = HTTPServer(server_address,CGIHTTPRequestHandler) # 创建服务器对象
        server_obj.serve_forever() # 启动服务器
    except OSError as e:
        print(e)

def run_http_server(path, port):
    p = multiprocessing.Process(target=http_server, args = (path, port))
    p.start()
    print(f"[{p.pid} / {port}] Run http server at " + path + '...')

if __name__ == "__main__":
    run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)