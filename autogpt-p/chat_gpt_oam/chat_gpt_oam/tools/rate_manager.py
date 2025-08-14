import collections
import threading
import time

from chat_gpt_oam.tools.singleton import Singleton


class RateManager(Singleton):

    def __init__(self, max_rate=200):
        if not hasattr(self, 'max_rate'):
            self.max_rate = max_rate
            self.mutex = threading.Semaphore()
            self.timestamps = []

    def wait_until_rate(self):
        with self.mutex:
            print("enter")
            time.sleep(0.1)
            current_time = time.time()
            first_entry_winin_min = 0
            while len(self.timestamps) >= self.max_rate:
                for i, t in enumerate(self.timestamps):
                    if current_time - 60 < t:
                        print("First: " +str(i))
                        first_entry_winin_min = i
                        break
                self.timestamps = self.timestamps[first_entry_winin_min:]
                current_time = time.time()
                time.sleep(0.5)
            self.timestamps.append(time.time())
            print("exit")


