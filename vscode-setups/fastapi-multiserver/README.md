# VS Code Setup for Multiserver FastAPI Projects

### This setup does the following:
1. Creates network interfaces for either MacOS or Linux (see `setup.sh`) to allow a client (see `client.py`) to run on IP adress `192.168.100.2`.
2. Uses compound configurations in `luanch.json` to set up the network interfaces (see the `preLaunchTask` in `launch.json` and the `Test Setup` task in `tasks.json`) launch both the client and server in debug mode. 
3. Perform clean up tasks after debugging is done (see `postDebugTask` in `launch.json` and `Test Cleanup` task in `tasks.json`). 

### How to run this example:
1. Place breakpoints in either `server.py` and/or `client.py`.
2. Open up the debugging tab on the left-hand side of VS Code editor and select and run the `FastAPI Test` configuration.
3. Open up a browser, navigate to either `http://192.168.100.2:8002/` or `http://127.0.0.1:8001/`.
4. Step through the debugger. 