#!/usr/bin/env python
# -*- coding:utf-8 -*-
import codecs
import subprocess
import time
import sys
import os
import settings
from utils import receive_param, log

# 启动后，自动执行爬取程序，直到数据爬取完成

TIME = 10 * 60  # 程序状态检测间隔（单位：秒钟）
CMD = "stock_crawler.py"  # 需要执行程序的绝对路径


class Auto_Run():
    def __init__(self, sleep_time, cmd):
        self.sleep_time = sleep_time
        self.cmd = cmd
        self.ext = (cmd[-2:]).lower()  # 判断文件的后缀名，全部换成小写
        self.p = None  # self.p为subprocess.Popen()的返回值，初始化为None
        self.run()  # 启动时先执行一次程序

        try:
            # 日志文件路径
            log_path = log.getLogPath()
            # 判断日志文件是否已经存在
            isExist = os.path.isfile(log_path)
            # 如果存在，是否有数据
            nonEmpty = False
            if isExist:
                nonEmpty = os.path.getsize(log_path)

            exit_flag = False
            while True:
                if nonEmpty:
                    f1 = codecs.open(log_path, 'r', settings.FILE_ENCODE)
                    for line in f1.readlines():
                        if '爬取完成!' == line[-6:-1]:
                            print("测到程序已运行完成,程序退出!")
                            self.p.kill()
                            exit_flag = True
                            break
                    f1.close()
                if exit_flag:
                    break
                time.sleep(sleep_time)  # 休息一定时间（秒钟），判断程序状态
                self.poll = self.p.poll()  # 判断程序进程是否存在，None：表示程序正在运行 其他值：表示程序已退出
                if self.poll is not None:
                    print("检测到程序未运行,程序启动!")
                    self.run()
                else:
                    # 每两小时自动重启程序一次
                    time.sleep(sleep_time * 12)
                    self.p.kill()
                    self.run()
        except KeyboardInterrupt:
            print("检测到Ctrl+C,程序退出!")
            self.p.kill()  # 检测到CTRL+C时，kill掉CMD中启动的exe或者jar程序
            sys.exit()

    def run(self):
        if self.ext == "py":
            print("执行 python %s" % self.cmd)
            self.p = subprocess.Popen(['python', '%s' % self.cmd], stdin=sys.stdin, stdout=sys.stdout,
                                      stderr=sys.stderr, shell=False)
        elif self.ext == "exe":
            print("启动%s程序" % self.cmd)
            self.p = subprocess.Popen('%s' % self.cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
                                      shell=False)


if __name__ == '__main__':
    if isinstance(receive_param.getArgs(), tuple):
        f2 = codecs.open('resources/date.txt', 'w', settings.FILE_ENCODE)
        # 写入
        w = lambda x: f2.write(x)
        data = receive_param.getArgs()[0] + receive_param.getArgs()[1]
        w(data)
        f2.close()

        Auto_Run(TIME, CMD)
    else:
        receive_param.getArgs()
