# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='MindsporeTrainer',
      version='0.1.0',
      description='Make Mindspore Training Easier',
      long_description=long_description,
      author='Zhou, Bo',
      author_email='baudzhou@outlook.com',
      url='https://github.com/baudzhou/MindsporeTrainer',
      license="MIT",
      classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
      ],
      install_requires=[
                        'loguru',
                        'ujson',
                        'tqdm',
                        'scipy',
                        'scikit-learn',
                        ],
      keywords='Deep Learning, NLP, CV, Transformers',
      packages=find_packages(),
      # package_dir={'mstrainer':'MindsporeTrainer'},
      # package_data={'':['*.py']}
)
