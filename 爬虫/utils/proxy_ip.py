#!/usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import random as rd
import settings

# 随机选取一个代理IP


def set_proxy():
    proxys = []
    f = codecs.open('proxy_ip.txt', 'r', settings.FILE_ENCODE)
    for line in f.readlines():
        proxy_host = line.strip()
        proxys.append(proxy_host)
    proxy = rd.choice(proxys)
    f.close()
    return proxy
