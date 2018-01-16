#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 获取一年份的url信息


def getURL(startYear, endYear, startMonth, endMonth, startDay, endDay):

    # 设置两个传入参数：股票代码 时间
    urlTemplate = "http://stock.gtimg.cn/data/index.php?appn=detail&action=download&c={0}&d={1}"

    # 年份
    if endYear - startYear == 0:
        for year in range(startYear, endYear + 1):
            strYear = str(year)
            for url, strYearMonthDay, stockCode in getMonthDay(urlTemplate, strYear, startMonth, endMonth,
                                                               startDay, endDay):
                yield url, strYearMonthDay, stockCode
    elif endYear - startYear == 1:
        for year in range(startYear, endYear + 1):
            if year == startYear:
                strYear = str(year)
                endMonth = 12
            else:
                strYear = str(year)
                startMonth = 1
            for url, strYearMonthDay, stockCode in getMonthDay(urlTemplate, strYear, startMonth, endMonth,
                                                               startDay, endDay):
                yield url, strYearMonthDay, stockCode
    else:
        for year in range(startYear, endYear + 1):
            if year == startYear:
                strYear = str(year)
                endMonth = 12
            elif startYear < year < endYear:
                strYear = str(year)
                startMonth = 1
                endMonth = 12
            else:
                strYear = str(year)
                startMonth = 1
            for url, strYearMonthDay, stockCode in getMonthDay(urlTemplate, strYear, startMonth, endMonth,
                                                               startDay, endDay):
                yield url, strYearMonthDay, stockCode


def getMonthDay(urlTemplate, strYear, startMonth, endMonth, startDay, endDay):
    # 月份：12个月
    for month in range(startMonth, endMonth + 1):
        if month < 10:
            strMonth = '0' + str(month)
            # 天数：31天
            for day in range(startDay, endDay + 1):
                if day < 10:
                    strDay = '0' + str(day)
                else:
                    strDay = str(day)
                # 某一天的年月日
                strYearMonthDay = strYear + strMonth + strDay
                # 获取股票代码
                # 设置股票代码文件所在路径
                filePath = 'resources/stock_code_alone.txt'
                f = open(filePath)
                for line in f.readlines():
                    # 单个股票对应代码
                    stockCode = line.strip('\n')
                    # 传入url参数
                    url = urlTemplate.format(stockCode, strYearMonthDay)
                    yield url, strYearMonthDay, stockCode
                f.close()
        else:
            strMonth = str(month)
            for day in range(startDay, endDay + 1):
                if day < 10:
                    strDay = '0' + str(day)
                else:
                    strDay = str(day)
                strYearMonthDay = strYear + strMonth + strDay
                f = open('resources/stock_code_alone.txt')
                for line in f.readlines():
                    stockCode = line.strip('\n')
                    url = urlTemplate.format(stockCode, strYearMonthDay)
                    yield url, strYearMonthDay, stockCode
                f.close()
