from setuptools import setup, find_packages


setup(
    name="hello_libary",
    version="0.0.1",
    author="Minh-Quan Do",
    author_email="mdo9@gmu.edu",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_data={
        '': ['__pycache__', '*.pyc', '*.pyo']
    },
    install_requires=[
        "fastapi"
    ]
)