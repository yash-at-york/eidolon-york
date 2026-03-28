import threading, time
from ghost_cli_v3 import SQLiteMapper

def worker(db):
    db.save_mapping('h_test', 'test_name')

mapper = SQLiteMapper()
thread = threading.Thread(target=worker, args=(mapper,))
thread.start()
thread.join()
mapper.close()
print('Threaded SQLite OK')
