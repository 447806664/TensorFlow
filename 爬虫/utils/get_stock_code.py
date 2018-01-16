#!/usr/bin/env python
# -*- coding:utf-8 -*-

import requests

from lxml import etree

# 爬取股票代码数据


url = 'http://quote.eastmoney.com/stocklist.html'
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:48.0) Gecko/20100101 Firefox/48.0'
}
proxy = {
    "http": "http://wsg.cmszmail.ad:8083/"
}

response = requests.get(url, headers=header, proxies=proxy)
html = response.content.decode('gbk')
selector = etree.HTML(html)


def getStockCode():
    contents = selector.xpath('//div[@id="quotesearch"]/ul/li')
    for content in contents:
        code = content.xpath('a/text()')[0][-7:-1]
        name = content.xpath('a/text()')[0][:-8]
        yield code, name


def writeStockCode():
    f = open('resources/StockCodeNew.txt', 'wb+')
    w = lambda x: f.write(x.encode('utf-8'))

    # 写入股票代码和名称
    # StockCode = getStockCode()
    # w('股票代码一览表\n\n')
    # for code, name in StockCode:
    #     w(code.replace(' ', '') + '\t')
    #     w(name.replace(' ', '') + '\n')

    # 写入股票类型和代码
    StockCode = getStockCode()
    for code, name in StockCode:
        w(code + '\n')

    f.close()

    print('爬取完成！')


if __name__ == '__main__':
    writeStockCode()
