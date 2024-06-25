How to run project:
```
$ docker compose up --build
```

The goal of this mini project is to show a simple set up of a python development environment using docker.

Features include:
- Dockerfile that copies the python package being developed into the docker container.
- Dockerfile that call `pip install -e <path of python package>` to install python package.
- Since testing folder is completely outside of the package, these tests will not be ran when publishing the package. These tests are intended to be ran during a CI/CD process.

Notes:
- Directory containing code for python package cannot be mounted into docker container. I suspect this is due to the fact that the command `pip install -e` creates a symlink from the site-packages directory to the package's source directory. Since the docker container has its own site-packages directory, creating symlinks with the package directory that's mounted into th container might cause issues. 