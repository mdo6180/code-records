from time import sleep
import cProfile
import pstats



def long_func():
    print("functioned entered")
    sleep(3)

    a = []
    for i in range(100000):
        a.append(i*i)


def waste_time():
    sleep(5)


def another_long_func():
    long_func()

    a = []
    for i in range(300000):
        a.append(i*i)
    
    waste_time()



with cProfile.Profile() as profile:
    another_long_func()

results = pstats.Stats(profile)
results = results.sort_stats(pstats.SortKey.TIME)
results.print_stats()
