#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: tian
@contact: nieliangtian@foxmail.com
@python: python3
@software: PyCharm Community Edition
@file: continue_and_break.py
@time: 2018/1/19 11:01
"""

a = True

while a:
    b = input('type something:')
    if b == '1':
        a = False
    else:
        # 占位语句, do nothing
        pass
    print('still in while')

print('finish run')
