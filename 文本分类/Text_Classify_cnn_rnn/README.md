基于 cnn、rnn的文本分类
python环境: 3.6
数据下载地址:链接:https://pan.baidu.com/s/1MZYmzt-4spX85mupXu-HTg 密码:ivbm
训练数据格式: 类别、新闻内容(也可以考虑 先将新闻内容生成摘要,对摘要进行训练)
config_utils 环境配置文件工具类
data_utils 数据预处理工具类
数据处理过程: 代码中有详细注释

主要遇到问题:
1.当新闻文本内容中文字过多时  维度过大可能会有内存溢出等情况，因此需要根据词频大小限制维度
2.模型加载时,若保存了参数更新次数 需要使用 tf.train.latest_checkpoint(checkpoint_path) 加载最后一次保存的模型


