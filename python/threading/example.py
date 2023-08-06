import multiprocessing
from multiprocessing import Value
import time

def infinite_loop1(resume_flag: Value):
    while True:
        try:
            if resume_flag.value == 0:
                print("Infinite loop1 is paused...")
                time.sleep(0.5)

            elif resume_flag.value == 1:
                print("Infinite loop1 is running...")

            elif resume_flag.value == 2:
                print("ending loop1 child process")
                break

            time.sleep(1)

        except KeyboardInterrupt:
            time.sleep(0.2)

def infinite_loop2(resume_flag):
    while True:
        try:
            if resume_flag.value == 0:
                print("Infinite loop2 is paused...")
                time.sleep(0.5)

            elif resume_flag.value == 1:
                print("Infinite loop2 is running...")

            elif resume_flag.value == 2:
                print("ending loop2 child process")
                break

            time.sleep(1)

        except KeyboardInterrupt:
            time.sleep(0.2)

if __name__ == "__main__":
    
    # Create a multiprocessing value to control the loop execution
    resume_flag = multiprocessing.Value('i', 1)  # 1 indicates loop is running, 0 indicates it's paused

    # Create a process with the infinite loop function
    process1 = multiprocessing.Process(target=infinite_loop1, args=(resume_flag,))
    process2 = multiprocessing.Process(target=infinite_loop2, args=(resume_flag,))

    # Start the process to execute the infinite loop
    process1.start()
    process2.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:

            # Pause the infinite loop
            print("\nPausing the infinite loop...")
            resume_flag.value = 0

            user_input = input("\nAre you sure you want to stop the pipeline? (yes/no) Press Enter to abort.")

            if user_input.lower() == "yes":

                user_input = input("Enter 'hard' for a hard stop, enter 'soft' for a soft stop? Press Enter to abort.")
                if user_input.lower() == "hard":
                    resume_flag.value = 2
                    process1.join()
                    process2.join()
                    break

                elif user_input.lower() == "soft":
                    print("\nExiting... tearing down all nodes in DAG")
                    time.sleep(2)
                    print("All nodes teardown complete")

                    resume_flag.value = 2
                    process1.join()
                    process2.join()
                    break
                
                else:
                    # Resume the infinite loop
                    print("Resuming the infinite loop...")
                    resume_flag.value = 1
                
            else:
                # Resume the infinite loop
                print("Resuming the infinite loop...")
                resume_flag.value = 1
