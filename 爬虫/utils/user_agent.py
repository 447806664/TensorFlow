#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random as rd

import settings

# 随机选取一个User-Agent


def set_headers():
    ua = rd.choice(settings.USER_AGENT_LIST)
    # print('User-Agent:', ua)
    return ua
