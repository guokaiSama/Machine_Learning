#!/usr/bin/python
# coding:utf-8
"""
SVD算法的具体应用
by:guoKaiSama
"""
import recomSVD
if __name__=="__main__":
    # 使用sklearn对矩阵进行SVD分解
    #recomSVD.sklearnSVD()
    # 基于协同过滤的推荐系统
    #recomSVD.recommendTest()
    # 压缩图片,允许基于任意给定的奇异值数目来重构图像
    print recomSVD.imgCompress(2)

