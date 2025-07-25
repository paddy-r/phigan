import multiprocessing as mp
import time


def worker(num):
    print(f'Worker {num} is starting')
    time.sleep(10)
    print(f'Worker {num} is done')
    return num * 2


if __name__ == '__main__':
    start = time.time()

    num_cores = mp.cpu_count()*10

    with mp.Pool(processes=num_cores) as pool:  # Create a pool of N worker processes
        results = pool.map(worker, range(num_cores))  # Map the worker function to the range

    print('Results:', results)

    end = time.time()
    elapsed = end - start
    print('All workers are done in {}s'.format(elapsed))
