# HR 30/04/25 Basic multiprocessing example to test speed improvements
# This covers only mp.Process; must also test (e.g.) Pool, map, imap and starmap
# as well as shared memory objects

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import uuid
import time
import ipf_scratch

DATA_DIR = ipf_scratch.DATA_DIR

# Create a large dataset
# global large_dataset
# large_dataset = np.random.rand(10000, 10000)

large_dataset = pd.read_csv(os.path.join(DATA_DIR, '2019_US_cohort.csv'))  # Basic Minos synthpop

# Parameters here, so outside of timings
n = mp.cpu_count() - 1
n_repeat = 20


def worker(data, n_repeat):
    for i in range(n_repeat):
        done = pd.crosstab(data['age'], data['ethnicity']).sum()
        done.count().sum()
    return done


if __name__ == '__main__':

    # Case 1: Single process:
    start1 = time.time()
    result1 = worker(large_dataset, n_repeat)
    end1 = time.time()
    elapsed1 = end1 - start1
    print("Time (single process): {}".format(elapsed1))

    # Case 2: Multiprocess with shared dataset
    start2 = time.time()

    # Create processes
    processes = []
    for _ in range(n):  # Create N processes
        p = mp.Process(target=worker, args=(large_dataset, n_repeat))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    end2 = time.time()
    elapsed2 = (end2 - start2) / n

    f1 = 1 / n
    f2 = elapsed2 / elapsed1
    print("Time per process: {}".format(elapsed2))
    print("Max. possible speedup factor: {}".format(f1))
    print("Actual speedup factor: {}".format(f2))
    print("Overall efficiency: {}".format(f1 / f2))
