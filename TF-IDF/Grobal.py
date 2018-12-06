# -*- coding: utf-8 -*-
import os

pattern = "full"  # 搜索模式:"full"为全文搜索模式,"keys"为关键词搜索模式
n = 10  # 关键词搜索时的关键词数量
ruler_list = []  # 不需要的字符

PreprocessResultName = "pro_res.txt"  # 预处理文件名
ResultFileName = "result.txt"  # 搜索结果文件名

base_path = os.getcwd() + os.sep
original_data_path = os.path.join(base_path+'data/original_data/') # 原始数据
title_and_abs_path = os.path.join(base_path+'data/title_and_abs/')  # 处理后的标题和摘要
pro_keyword_path = os.path.join(base_path+'data/pro_keyword/')  # 预处理的关键词存放目录数据
keyword_path = os.path.join(base_path+'data/keyword/')  # 关键词存放目录数据

