import threading
import time
from typing import Dict



# Define color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Reset to default color



class RootNode(threading.Thread):
    def __init__(self, name: str, task_name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.task_name = task_name
        self.shutdown_event = threading.Event()
        self.successors: Dict[str, 'LeafNode'] = {}
        self.events_tables: Dict[str, threading.Event] = {}
        self.pause_event = threading.Event()
        self.pause_event.set()
    
    def add_successor(self, node: 'LeafNode'):
        self.successors[node.name] = node
        self.events_tables[node.name] = threading.Event()

    # called by the successor to signal the root node
    def signal_predecessor(self, successor_name: str):
        self.events_tables[successor_name].set()
    
    def shutdown(self):
        print(f"Shutting down {self.name}...")
        for event in self.events_tables.values():
            event.set()
        self.pause_event.set()
        self.shutdown_event.set()
    
    def pause(self):
        self.pause_event.clear()
    
    def resume(self):
        self.pause_event.set()
    
    def wait_successors(self):
        for event in self.events_tables.values():
            event.wait()

    def run(self):
        while self.shutdown_event.is_set() is False:
            # clear all events
            for event in self.events_tables.values():
                event.clear()
            
            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            # execute the task
            print(f"{Colors.OKGREEN}Thread {self.name} {Colors.WARNING}executing{Colors.OKGREEN} task {self.task_name}{Colors.ENDC}")
            time.sleep(1)

            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            # signal all successors
            for successor in self.successors.values():
                print(f"{Colors.OKGREEN}Thread {self.name} {Colors.FAIL}signaling{Colors.OKGREEN} {successor.name}{Colors.ENDC}")
                successor.signal_successor(self.name)
            
            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            # wait for all successors to finish
            self.wait_successors()

            self.pause_event.wait()
            if self.shutdown_event.is_set(): break


class LeafNode(threading.Thread):
    def __init__(self, name: str, task_name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.task_name = task_name
        self.shutdown_event = threading.Event()
        self.predecessors: Dict[str, RootNode] = {}
        self.events_tables: Dict[str, threading.Event] = {}
        self.pause_event = threading.Event()
        self.pause_event.set()

    def add_predecessor(self, node: RootNode):
        self.predecessors[node.name] = node
        self.events_tables[node.name] = threading.Event()
    
    def signal_successor(self, node_name: str):
        self.events_tables[node_name].set()
    
    def shutdown(self):
        print(f"Shutting down {self.name}...")
        for event in self.events_tables.values():
            event.set()
        self.pause_event.set()
        self.shutdown_event.set()

    def pause(self):
        self.pause_event.clear()
    
    def resume(self):
        self.pause_event.set()

    def run(self):
        while self.shutdown_event.is_set() is False:
            
            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            # wait for all predecessors to finish
            for event in self.events_tables.values():
                event.wait()

            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            print(f"{Colors.OKBLUE}Thread {self.name} {Colors.WARNING}executing{Colors.OKBLUE} task {self.task_name}{Colors.ENDC}")
            time.sleep(1)

            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            # signal all predecessors
            for predecessor in self.predecessors.values():
                print(f"{Colors.OKBLUE}Thread {self.name} {Colors.FAIL}signaling{Colors.OKBLUE} {predecessor.name}{Colors.ENDC}")
                predecessor.signal_predecessor(self.name)

            self.pause_event.wait()
            if self.shutdown_event.is_set(): break

            # clear all events
            for event in self.events_tables.values():
                event.clear()


if __name__ == "__main__":
    node1 = RootNode(name="node1", task_name="task1")
    node2 = RootNode(name="node2", task_name="task2")
    node3 = LeafNode(name="node3", task_name="task3")

    try:
        node1.add_successor(node3)
        node2.add_successor(node3)
        node3.add_predecessor(node1)
        node3.add_predecessor(node2)
        node1.start()
        node2.start()
        node3.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        node1.shutdown()
        node2.shutdown()
        node3.shutdown()
        node1.join()
        node2.join()
        node3.join()
        print("All nodes have been shut down.")