#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import sys
import getopt


# 获取运行Python脚本时传入的参数


def getArgs():
    param1 = ''

    try:
        if len(sys.argv[1:]) == 4:
            opts, args = getopt.getopt(sys.argv[1:], 's:e:', ['start=', 'end='])
        else:
            print('请在".py"后空一格,输入数据爬取开始日期和结束日期!\n'
                  '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                  '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
            sys.exit()
    except getopt.GetoptError:
        print('输入有误!\n'
              '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
              '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
        sys.exit()

    flag = False
    for opt, param in opts:
        if opt in ('-s', '--start'):
            if len(param) == 8:
                param1 = param
                flag = True
            else:
                print('输入有误!\n'
                      '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                      '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
                sys.exit()

        if flag:
            if opt in ('-e', '--end'):
                if len(param) == 8:
                    param2 = param
                    if ckeckParam(param1, param2):
                        return param1, param2
                else:
                    print('输入有误!\n'
                          '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                          '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
                    sys.exit()
        else:
            print('输入有误!\n'
                  '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                  '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
            sys.exit()


# 检查接收参数的有效性
def ckeckParam(param1, param2):
    start_year = param1[0:4]
    end_year = param2[0:4]
    start_month = param1[4:6]
    end_month = param2[4:6]
    start_day = param1[6:]
    end_day = param2[6:]

    if re.match('^[12][0-9]{3}$', start_year) and \
            re.match('^[12][0-9]{3}$', end_year) and \
            re.match('^[01][0-9]$', start_month) and \
            re.match('^[01][0-9]$', end_month) and \
            re.match('^[0-3][0-9]$', start_day) and \
            re.match('^[0-3][0-9]$', end_day):
        start_year = int(start_year)
        end_year = int(end_year)
        start_month = int(start_month)
        end_month = int(end_month)
        start_day = int(start_day)
        end_day = int(end_day)
        if start_year > end_year:
            print('开始年份不能大于结束年份!\n'
                  '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                  '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
            sys.exit()
        else:
            if start_month > end_month:
                print('开始月份不能大于结束月份!\n'
                      '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                      '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
                sys.exit()
            elif start_month < 0 or start_month > 31 or end_month < 0 or end_month > 32:
                print('输入的月份不对!\n'
                      '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                      '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
                sys.exit()
            else:
                if start_day > end_day:
                    print('开始日期不能大于结束日期!\n'
                          '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                          '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
                    sys.exit()
                elif start_day < 0 or start_day > 31 or end_day < 0 or end_day > 32:
                    print('输入的日期不对!\n'
                          '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
                          '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
                    sys.exit()
                else:
                    return True
    else:
        print('输入的日期格式不对!\n'
              '参照:python auto_run.py --start <yyyyMMdd> --end <yyyyMMdd>\n'
              '例如:python auto_run.py --start 20170101 --end 20170131 表示2017年一整月')
        sys.exit()


if __name__ == '__main__':
    start_year = int(getArgs()[0][0:4])
    end_year = int(getArgs()[1][0:4])
    start_month = int(getArgs()[0][4:6])
    end_month = int(getArgs()[1][4:6])
    start_day = int(getArgs()[0][6:])
    end_day = int(getArgs()[1][6:])
    print('start_year ', start_year)
    print('end_year ', end_year)
    print('start_month ', start_month)
    print('end_month ', end_month)
    print('start_day ', start_day)
    print('end_day', end_day)
