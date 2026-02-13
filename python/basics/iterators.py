def my_iterator(items):
    print(">>> Starting iteration")

    for item in items:
        yield item

    print(">>> Finished iteration")


data = [1, 2, 3]

for x in my_iterator(data):
    print(x)
