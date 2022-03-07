from time import time

timings = []

class Timer(object):
    def __enter__(self):
        self.start = time()
    def __exit__(self, type, value, traceback):
        self.end = time()
        timings.append(self.end - self.start)
