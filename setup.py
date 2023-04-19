# Copyright (c) Alibaba, Inc. and its affiliates.
from setuptools import setup, find_packages

setup(
    name='llm-evals',
    version='0.0.1',
    author='xingjun.wxj',
    author_email='xingjun.wxj@alibaba-inc.com',
    description='LLMs Evals framework',
    url='',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

# python3 setup.py sdist bdist_wheel
