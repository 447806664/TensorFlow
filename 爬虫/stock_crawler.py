#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time

from utils import stock_data

'''
 启动程序，直接运行即可
 无错误提示即表示程序运行正常
 若异常退出,请重启程序,数据会从写入失败的地方接着写入
 默认爬取2017年1月份整月的数据
 相关参数在settings中变更
'''


if __name__ == '__main__':
    print('****************** 股票数据爬取系统 ******************')
    time.sleep(1)
    stock_data.writeFile()
