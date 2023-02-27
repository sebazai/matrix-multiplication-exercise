
import random
from threading import Thread
import time
import statistics

def main():
  maxN = 1000
  numArray = random_array(maxN, 1, 10000)
  time_samples = []
  for i in range(1000):
    start_time = get_current_time()
    N = random.randint(1, maxN)
    thread = Thread(target=thread_function, args=(numArray, N))
    thread.start()
    thread.join()
    end_time = get_current_time()
    time_samples.append(end_time - start_time)
  summary_stats = (min(time_samples), max(time_samples), statistics.mean(time_samples), statistics.stdev(time_samples))
  print("Minimum: {}\nMaximum: {}\nAverage: {}\nStandard deviation: {}".format(*summary_stats))

def thread_function(numArray, N):
  sum(numArray, N)
  return

def random_array(N, min, max):
  # create an array of N random number in the range (min, max) and return it.
  array = []
  for i in range(N):
    array.append(random.randint(min, max))
  return array

def get_current_time():
  # return the current time in milliseconds.
  return time.time()

if __name__ == "__main__":
  main()