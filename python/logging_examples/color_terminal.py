import time
from datetime import timedelta


def color_log(start_msg, end_msg, color):
    colors = {
        "HEADER": "\033[95m",
        "OKBLUE": "\033[94m",
        "OKCYAN": "\033[96m",
        "OKGREEN": "\033[92m",
        "WARNING": "\033[93m",
        "FAIL": "\033[91m",
        "ENDC": "\033[0m"
    }

    if color not in colors:
        colors_list = [key for key in colors.keys()]
        colors_list = ", ".join(colors_list)
        raise ValueError(
            "Color is not a valid color name! Valid color names are:\n{}".format(colors_list)
        )

    def decorator(func):

        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            print(
                "{}{}...{}".format(
                    colors[color],
                    start_msg,
                    colors["ENDC"]
                )
            )

            result = func(*args, **kwargs)

            end_time = time.monotonic()
            elapsed_time = end_time - start_time
            print(
                "{}{}. Elapsed time = {} {}".format(
                    colors[color],
                    end_msg,
                    str(timedelta(seconds=end_time-start_time))[:-3],
                    colors["ENDC"]
                )
            )

            return result

        return wrapper

    return decorator


if __name__ == "__main__":

    @color_log("hello", "goodbye", "FAIL")
    def testing():
        time.sleep(1.5)

    testing()
