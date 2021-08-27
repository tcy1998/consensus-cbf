from multiprocessing.pool import ThreadPool as Pool
from os import getpid
import time
import pandas as pd


pyfiles = [10,2,3,5]    

def scraper(x):
    results_df = pd.DataFrame({})
    print('Program started:',x,"I'm process", getpid())
    time.sleep(x)
    print('Program completed:',x)
    results_df.to_csv('multi{}.csv'.format(x))


if __name__ == '__main__':
    with Pool(4) as pool:
        start=time.time()
        result = pool.map(scraper, pyfiles)
        pool.terminate()
        pool.join()
        print("Time Taken: ",str(time.time()-start))