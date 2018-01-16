#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import getopt


# 测试运行时传入参数是否有效


def getArgs():
    param1 = ''
    opts = ''

    try:
        if len(sys.argv[1:]) >= 4:
            opts, args = getopt.getopt(sys.argv[1:], 's:e:')
        else:
            print('输入有误!请参照:python param_transmit.py -s <yyyyMMdd> -e <yyyyMMdd>')
    except getopt.GetoptError:
        print('输入有误!请参照:python param_transmit.py -s <yyyyMMdd> -e <yyyyMMdd>')
        sys.exit()

    flag1 = False
    for opt, param in opts:
        if opt == '-s':
            if len(param) == 6:
                param1 = param
                flag1 = True
            else:
                print('输入有误!请参照:python param_transmit.py -s <yyyyMMdd> -e <yyyyMMdd>')
                break

        if flag1:
            if opt == '-e':
                if len(param) == 6:
                    param1 = param1
                    param2 = param
                    return param1, param2
                else:
                    print('输入有误!请参照:python param_transmit.py -s <yyyyMMdd> -e <yyyyMMdd>')
                    break
        else:
            print('输入有误!请参照:python param_transmit.py -s <yyyyMMdd> -e <yyyyMMdd>')
            break


if __name__ == '__main__':
    print(getArgs())
