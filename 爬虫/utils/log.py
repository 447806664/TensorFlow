#!/usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import logging
import settings

# 日志打印


def getLogPath():
    # 读取从命令行接收的参数
    f = codecs.open('resources/date.txt', 'r', settings.FILE_ENCODE)

    for line in f.readlines():
        # 设置爬取数据写入路径
        logPath = 'logs/log_' + \
                   line[0:4] + line[4:6] + line[6:8] + '_' + \
                   line[8:12] + line[12:14] + line[14:]
        return logPath

    f.close()


logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(getLogPath(), 'a', encoding=settings.FILE_ENCODE)
fh.setLevel(logging.INFO)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)
