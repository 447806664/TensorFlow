#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: try_except.py
@time: 2018/1/19 11:10
"""

try:
    file = open('data/word.txt', 'r+')
except Exception as e:
    print(e)
    response = input('do you want to create a new file:')
    if response == 'y':
        file = open('data/word.txt', 'w')
        file.write('This is a demo.')
        file.close()
    else:
        pass

