# -*- coding: utf-8 -*-

from __future__ import division
import re
import jieba
import math
# 这个函数用于预处理文件处理过程中采用unicode编码
import os
import Grobal


def fullcut(content):
    cut_content = jieba.cut(content, cut_all=False)
    word_list_temp = list(cut_content)
    word_list = []
    if not Grobal.ruler_list:
        r = r'[^/]{2,}'
        temp = '/'.join(word_list_temp)
        word_list = re.findall(r, temp)
    else:
        for word in word_list_temp:
            if word not in Grobal.ruler_list:
                word_list.append(word)
    return word_list


def halfcut(content):
    word_list = []
    k = Grobal.n
    while True:
        cut_content = jieba.analyse.extract_tags(content, k)
        word_list_temp = cut_content
        if not Grobal.ruler_list:
            r = r'[^/\d]{2,}'
            temp = '/'.join(word_list_temp)
            word_list = re.findall(r, temp)
        else:
            for word in word_list_temp:
                if word not in Grobal.ruler_list:
                    word_list.append(word)
                    # print len(word_list)
        if (len(word_list) >= Grobal.n):
            break
        else:
            word_list = []
            k += 1
    return word_list

# 多字符串替换函数，对于str_source中的某些字符（从*words传入）用char代替
def str_replace(str_source, char, *words):
    str_temp = str_source
    for word in words:
        str_temp = str_temp.replace(word, char)
    return str_temp

# 将所有文本分词，结果汇总到pro_res.txt
def prepro_file(fl_in_url, re_out_url, *wd_be_del):
    in_url = fl_in_url.replace('\\', '/')
    out_url = re_out_url.replace('\\', '/')
    fl_in = os.listdir(in_url)
    re_out = open(out_url, 'w',encoding='utf8')
    i = 0
    for file in fl_in:
        i += 1
        afile_url = fl_in_url + '/' + file
        if os.path.isfile(afile_url):
            lines = open(afile_url, "r",encoding='utf8').readlines()
            content_temp = "".join(lines)
            if not wd_be_del:
                content = str_replace(content_temp, "", "\t", "\n", " ")  # 删除某些特殊字符如\t,\n等以保证是一行的连续的
            else:
                content = str_replace(content_temp, '', *wd_be_del)
            if Grobal.pattern == "full":
                cut_result = fullcut(content)
            else:
                cut_result = halfcut(content)
            out_str = ','.join(cut_result)
            re_out.write(file+'\t' + out_str+'\n')


if __name__ == '__main__':
    str = fullcut('我今天真的是很卡机的拉克丝多放几列卡萨丁金佛IE减肥了可视对讲发啦上课的放假阿里斯顿扣几分了深度光流口水的附近拉上课的放假')
    print(str)