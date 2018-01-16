#!/usr/bin/env python
# -*- coding:utf-8 -*-
import codecs
import os
import time
from urllib import request, error
import settings
from utils import web_url, proxy_ip, user_agent, log

'''爬取股票数据并写入本地文件'''


def getStockMarket(url):
    # 伪装
    headers = ("User-Agent", user_agent.set_headers())
    # # 代理
    # # True表示启用代理IP，False表示不启用代理IP
    # enable_proxy = settings.USE_PROXY
    #
    # proxyAddr = proxy_ip.set_proxy()
    # # 添加代理IP信息
    # proxy = request.ProxyHandler({settings.PROXY_TYPE: proxyAddr})
    # if enable_proxy:
    #     opener = request.build_opener(proxy)
    # else:
    #     opener = request.build_opener()
    opener = request.build_opener()

    # 添加浏览器伪装信息
    opener.addheaders = [headers]
    request.install_opener(opener)

    # 读取数据
    # 访问一个网页30s无响应，就重新访问
    # 打印日志
    log.logger.info('抓取网页 ' + url)
    try:
        data = request.urlopen(url, timeout=settings.TIMEOUT).read().decode(settings.WEBSITE_ENCODE)
    except error.URLError:
        data = request.urlopen(url, timeout=settings.TIMEOUT).read().decode(settings.WEBSITE_ENCODE)

    # 过滤掉由于URL不正确或者对应股票代码无数据导致的空值
    if data != '暂无数据':
        data = data.split(settings.SEPARATOR)
        data = data[1:2]
    else:
        data = ''
    return data


'''
爬取数据写入本地文件
文件编码方式：UTF-8
url为爬取目标网址链接
stockCode为股票代码
'''


def writeFile():
    # 写入操作开始时间
    startTime = time.time()

    '''
    数据爬取日期设定
    '''
    startYear =0
    endYear = 0
    startMonth = 0
    endMonth = 0
    startDay = 0
    endDay = 0

    # 读取从命令行接收的参数
    f1 = codecs.open('resources/date.txt', 'r', 'utf-8')

    # 爬取数据写入路径
    filePath = ''

    for line in f1.readlines():
        # 起止年份
        startYear = int(line[0:4])
        endYear = int(line[8:12])
        # 起止月份
        startMonth = int(line[4:6])
        endMonth = int(line[12:14])
        # 起止天数
        startDay = int(line[6:8])
        endDay = int(line[14:])

        # 设置爬取数据写入路径
        filePath = 'storage/stock_' + \
                   line[0:4] + line[4:6] + line[6:8] + '_' + \
                   line[8:12] + line[12:14] + line[14:]

    f1.close()

    # 判断爬取数据文件是否已经存在
    isExist = os.path.isfile(filePath)
    # 如果存在，是否有数据
    nonEmpty = False
    if isExist:
        nonEmpty = os.path.getsize(filePath)
    # 股票数据是否写入
    enable_write = False

    '''获取最后一行数据'''
    lastYearMonthDay, lastStockCode, lastData = '', '', ''

    # 如果文件已有数据，读取最后一行数据
    if nonEmpty:
        f2 = codecs.open(filePath, 'r', settings.FILE_ENCODE)
        # 初始偏移量
        off = -10
        while True:
            # 从文件末尾开始，向前10个字符
            f2.seek(off, 2)
            lines = f2.readlines()
            if len(lines) >= 2:
                lastLine = lines[-1]
                break
            off *= 2
        # 最后一行数据写入时的年月日
        lastYearMonthDay = lastLine[9:17]
        # 最后一行数据写入时的股票代码
        lastStockCode = lastLine[0:8]
        # 最后一行数据写入时的数据
        lastData = lastLine[18:].strip(settings.SEPARATOR)

        # 重新设定开始爬取数据的日期
        startYear = int(lastYearMonthDay[0:4])
        startMonth = int(lastYearMonthDay[4:6])
        startDay = int(lastYearMonthDay[6:])

        f2.close()

    # 获取url,strYearMonthDay,stockCode
    for url, strYearMonthDay, stockCode in web_url.getURL(startYear, endYear, startMonth, endMonth, startDay, endDay):
        # 获取股票数据
        stockMarket = getStockMarket(url)
        if stockMarket != '':
            # 设置文件写入方式及编码(追加写入)
            f3 = codecs.open(filePath, 'a+', settings.FILE_ENCODE)
            # 按行写入
            w = lambda x: f3.writelines(x)
            if nonEmpty:
                # 读取股票数据，和最后一行数据比对，直到匹配成功，才开始写入数据
                for data in stockMarket:
                    if enable_write:
                        # 写入字段：股票代码 + 时间 + 股票数据
                        data = strYearMonthDay + '\t' + stockCode + '\t' + data + '\n'
                        w(data)
                    if strYearMonthDay == lastYearMonthDay and stockCode == lastStockCode and data == lastData:
                        enable_write = True
            else:
                for data in stockMarket:
                    # 写入字段：股票代码 + 时间 + 股票数据
                    data = stockCode + '\t' + strYearMonthDay + '\t' + data + '\n'
                    w(data)
            f3.close()

    log.logger.info('爬取完成!')

    # 写入操作结束时间
    endTime = time.time()

    runTime = endTime - startTime

    # 时间输出格式化
    if runTime < 60:
        runTime = round(runTime, 2)
        log.logger.info('运行时间: ' + str(runTime) + '秒')
    elif runTime < 3600:
        runTimeMinute = runTime // 60
        runTimeMinute = int(runTimeMinute)
        runTimeSecond = runTime - runTimeMinute * 60
        runTimeSecond = round(runTimeSecond, 2)
        log.logger.info('运行时间: ' + str(runTimeMinute) + '分 ' + str(runTimeSecond) + '秒')
    else:
        runTimeHour = runTime // 3600
        runTimeHour = int(runTimeHour)
        runTimeMinute = (runTime - runTimeHour * 3600) // 60
        runTimeMinute = int(runTimeMinute)
        runTimeSecond = runTime - runTimeHour * 3600 - runTimeMinute * 60
        runTimeSecond = round(runTimeSecond, 2)
        log.logger.info('运行时间: ' + str(runTimeHour) + '小时 ' + str(runTimeMinute) + '分 ' + str(runTimeSecond) + '秒')
