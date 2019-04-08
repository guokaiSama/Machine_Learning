#!/usr/bin/python
# coding:utf-8
"""
朴素贝叶斯的算法实现

by:guoKaiSama
"""
import bayes

if __name__=="__main__":
    # 构建一个快速过滤器来屏蔽在线社区留言板上的侮辱性言论。
    # 如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标识为内容不当。
    #bayes.testingNB()

    # 使用朴素贝叶斯过滤垃圾邮件
    #bayes.spamTest()

    # sklearn的实现
    bayes.sklearnTest()
    #sklearnLogistic.func()

