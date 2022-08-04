import logging
import multiprocessing
import os
import time
from http.server import CGIHTTPRequestHandler, HTTPServer

from config.constants import DATASET_ROOT_DIR, DATASET_SERVER_PORT

logger = logging.getLogger('app')

def http_server(path, port):
    try:
        os.chdir(path)
        server_address = ("",port) # 设置服务器地址
        server_obj = HTTPServer(server_address,CGIHTTPRequestHandler) # 创建服务器对象
        server_obj.serve_forever() # 启动服务器
    except BlockingIOError as e:
        logger.error(f"BlockingIOError: {e}")
        time.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"Stopping Media Server...")
        server_obj.socket.close()
        server_obj.shutdown()
        logger.info(f"Media Server stopped.")
    except Exception as e:
        logger.exception(e)
        return

def run_http_server(path, port):
    p = multiprocessing.Process(target=http_server, args = (path, port))
    p.start()
    logger.info(f"[PID: {p.pid}] Media server started at {path}:{port}")

if __name__ == "__main__":
    run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)
