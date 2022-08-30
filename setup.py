# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(name='MindsporeTrainer',
      version='0.1.1',
      description='Make Mindspore Training Easier',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Zhou Bo',
      author_email='baudzhou@outlook.com',
      url='https://github.com/baudzhou/MindsporeTrainer',
      license="MIT",
      classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Text Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
      ],
      install_requires=[
                        'loguru',
                        'ujson',
                        'tqdm',
                        'scipy',
                        'scikit-learn',
                        ],
      keywords='Deep Learning, NLP, CV, Transformers, MindSpore',
      packages=find_packages(),
)
