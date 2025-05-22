import cProfile

profiler = cProfile.Profile()
profiler.enable()

# 🔍 Code block you want to profile
for i in range(1000000):
    _ = i ** 2

profiler.disable()
profiler.dump_stats("profile_results.prof")

# visualize the profile
# snakeviz profile_results.prof