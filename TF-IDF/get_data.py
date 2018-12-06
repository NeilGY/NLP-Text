# encoding:utf-8

import os
import sys
from bs4 import BeautifulSoup
import os
import Grobal
import re

filelist = os.listdir(Grobal.original_data_path)  # 取得当前路径下的所有文件

# 清洗出xml格式的文件中的标题和摘要信息
def get_text():
    abstracts = []
    for files in filelist:
        filename = os.path.splitext(files)[0]  # 取文件名
        soup = BeautifulSoup(open(Grobal.original_data_path + filename + '.xml',encoding='utf8'), 'html.parser')  # 解析网页
        b = soup.find("p", class_="abstracts")  # 取得"p", class_="abstracts"为标签的内容
        # print b
        if b is None or b.string is None:
            continue
        else:
            abstracts.extend(soup.title.stripped_strings)
            s = b.string
            abstracts.extend(s.lstrip(" "))
            f = open(Grobal.title_and_abs_path + filename + ".txt", "w+",encoding='utf8')  # 写入txt文件
            for i in abstracts:
                f.write(i)
            f.close()
            abstracts = []

        # getPro_keyword，清洗出xml文件中dl标签中的文本信息
        links = soup.find_all("dl")
        pro_f = open(Grobal.pro_keyword_path + filename + ".txt", "w+", encoding='utf8')# 将得到的未处理的文字放在pro_keyword文件夹中
        # print links
        for link in links:
            s1 = link.get_text()
            pro_f.write(s1)
        pro_f.close()


# 对上一步得到的getPro_keyword文件夹中的文件进行进一步处理，得到每个文件的关键字
def get_keyword():
    # getKeyword
    filelist = os.listdir(Grobal.pro_keyword_path)
    for files in filelist:
        filename = os.path.splitext(files)[0]
        is_key_word = False
        begin = 100000
        end = 10000
        f1 = open(Grobal.pro_keyword_path + filename + ".txt", "r",encoding='utf8')
        f2 = open(Grobal.keyword_path + filename + '.txt', "w+",encoding='utf8')
        for (num, value) in enumerate(f1):
            if value.count("基金项目") > 0 or value.count("机标分类号") > 0 or value.count("机标关键词") > 0 or value.count(
                    "基金项目") > 0 or value.count("DOI") > 0:
                is_key_word = False
            if is_key_word:
                value = value.strip()
                if value:
                    f2.write(value+",")
            if value.count("关键词") > 0:
                is_key_word = True
        f1.close()
        f2.close()


if __name__ == '__main__':
    get_text()
    get_keyword()
