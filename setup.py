# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

LONGDOC = """
MindsporeTrainer
=====

基于昇思MindSpore的训练框架


GitHub: https://github.com/baudzhou/MindsporeTrainer

特点
====

1.	采用纯python实现，方便多卡训练过程中的调试
2.	易于扩展，对新任务采用插件式接入
3.	方便实现多种模型的训练、评估、预测等


安装说明
========

推荐使用Python 3.7.x，在mindspore 1.5上测试通过

"""

setup(name='MindsporeTrainer',
      version='0.1.0',
      description='Make Mindspore Training Easier',
      long_description=LONGDOC,
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
