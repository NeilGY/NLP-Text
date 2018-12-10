# -*- coding:utf-8 -*-

import tensorflow as tf
from config_utils import make_path,clean,get_logger,load_config,save_config
from data_utils import build_vocab,read_vocab,read_category,read_file,process_file,batch_iter,clean_str
from model import Model
from collections import OrderedDict
import os
import time
from datetime import timedelta


#数据的路径
tf.flags.DEFINE_string("train_file", os.path.join('data','cnews.val.txt'), "Data source for the train data.")
tf.flags.DEFINE_string("test_file", os.path.join('data','cnews.test.txt'), "Data source for the test data.")
tf.flags.DEFINE_string("val_file", os.path.join('data','cnews.val.txt'), "Data source for the valid data.")
tf.flags.DEFINE_string("vocab_file", os.path.join('data/cnews.vocab.txt'), "vocabe_file for data")
tf.flags.DEFINE_string("word_to_id_file", os.path.join('data/cnews.word_to_id.txt'), "vocabe_file for data")
tf.flags.DEFINE_string("category_to_id_file", os.path.join('data/cnews.category_to_id.txt'), "vocabe_file for data")
tf.flags.DEFINE_string("log_path", 'log', "")
tf.flags.DEFINE_string("config_path", 'config', "")
#模型参数
tf.flags.DEFINE_integer("embedding_size", 120, "Dimensionality of character embedding (default: 120)")
#模型类型
tf.flags.DEFINE_string("model_type", "rnn", "model type in (cnn,rnn)")
#rnn
tf.flags.DEFINE_integer("lstm_dim", 100, "num of lstm")

#指定三个四个或者五个单词卷积
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#每次卷积得到128个特征图
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
#dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#学习率
tf.flags.DEFINE_float("lr", 0.001, "learning rate (default:0.5)")
#L2的惩罚项
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")

# Training parameters  训练参数

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
#评估
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#保存
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("checkpoints_path", 'result', "the path of checkpoints")

tf.flags.DEFINE_boolean("is_train", True, "train model")
tf.flags.DEFINE_boolean("is_clean", True, "clean train model")

FLAGS = tf.app.flags.FLAGS


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# config for the model
def config_model(seq_len,vocab_size,num_classes,filter_sizes):
    config = OrderedDict()
    config["seq_len"] = seq_len
    config["vocab_size"] = vocab_size
    config["num_classes"] = num_classes
    config["vocab_file"] = FLAGS.vocab_file
    config["log_path"] = FLAGS.log_path
    config["embedding_size"] = FLAGS.embedding_size
    config["model_type"] = FLAGS.model_type
    config["lstm_dim"] = FLAGS.lstm_dim
    config["filter_sizes"] = filter_sizes
    config["num_filters"] = FLAGS.num_filters
    config["dropout_keep_prob"] = FLAGS.dropout_keep_prob
    config["lr"] = FLAGS.lr
    config["l2_reg_lambda"] = FLAGS.l2_reg_lambda
    config["batch_size"] = FLAGS.batch_size
    config["num_epochs"] = FLAGS.num_epochs
    config["evaluate_every"] = FLAGS.evaluate_every
    config["num_checkpoints"] = FLAGS.num_checkpoints
    config["checkpoints_path"] = FLAGS.checkpoints_path
    config["train_file"] = FLAGS.train_file
    config["test_file"] = FLAGS.test_file
    config["val_file"] = FLAGS.val_file
    config["category_to_id_file"] = FLAGS.category_to_id_file
    config["word_to_id_file"] = FLAGS.word_to_id_file

    return config



def get_train_config():
    train_contents, train_labels = read_file(FLAGS.train_file)
    # 1.先构建训练数据的词汇字典
    if not os.path.exists(FLAGS.vocab_file):
        words = build_vocab(train_contents, FLAGS.vocab_file)
    else:
        words, _ = read_vocab(FLAGS.vocab_file)
    # 2.获取分类数据,构建分类数据的字典表,并保存至文件中
    categories, cat_to_id = read_category()
    # 3.生成训练配置文件
    vocab_size = len(words)
    num_classes = len(categories)
    #长度太大会内存溢出
    # seq_len = max([len(content) for content in train_contents])
    seq_len = 600
    filter_sizes = [int(i) for i in FLAGS.filter_sizes.split(',')]
    # 生成环境配置文件
    make_path(FLAGS)
    config_path = os.path.join(FLAGS.config_path, 'config')
    if not os.path.isfile(config_path):
        train_config = config_model(seq_len, vocab_size, num_classes, filter_sizes)
        save_config(train_config, config_path)
    return train_config


def train():
    train_config = get_train_config()
    _, word_to_id = read_vocab(train_config['vocab_file'])
    _, cat_to_id = read_category()
    # 获得日志
    logger = get_logger(os.path.join(FLAGS.log_path, 'train.log'))
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    with tf.Session(config=config) as sess:
        # 创建模型
        model = Model(train_config)
        # 获取数据训练
        x_train_data,y_train_data = process_file(train_config['train_file'],word_to_id,cat_to_id,train_config['seq_len'])
        #获取验证数据集
        x_val_data,y_val_data = process_file(train_config['val_file'],word_to_id,cat_to_id,train_config['seq_len'])
        #初始化变量
        sess.run(tf.global_variables_initializer())

        len_data = len(y_train_data)#数据样本数量
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
        flag = False #是否结束训练
        logger.info()
        #num_epochs:防止前面的学习丢失了一些特征,需要重复学习样本
        for i in range(train_config['num_epochs']):
            for x_input,y_output in batch_iter(x_train_data,y_train_data,train_config['batch_size']):
                total_batch += 1

                step, acc, loss = model.run_step(sess,x_input,y_output)
                #迭代100次评估一次模型
                if(step % FLAGS.evaluate_every == 0):
                    time_dif = get_time_dif(start_time)
                    logger.info("train: iterator{}: step:{}/{} acc:{} loss:{} time:{}".format(i+1,step%len_data,len_data,acc,loss,time_dif ))
                    val_acc, text_los = model.text_step(sess,x_val_data,y_val_data)
                    logger.info("test: acc:{} loss:{} ".format(acc,loss ))
                    #保存模型
                    if acc>0.5 and val_acc > 0.5 and val_acc > best_acc_val:
                        last_improved = total_batch
                        best_acc_val = val_acc
                        checkpoint_path = os.path.join(FLAGS.checkpoints_path,'checkpoints')
                        saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.max_to_keep)
                        saver.save(sess,checkpoint_path,global_step=step)
                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环

            if flag:
                time_dif = get_time_dif(start_time)
                logger.info('训练结束:{}'.format(time_dif))
                break


def evaluate_line():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(FLAGS.checkpoints_path, 'checkpoints')
    saver.restore(sess=session, save_path=checkpoint_path)  # 读取保存的模型
    config_path = os.path.join(FLAGS.config_path, 'config')
    test_config = load_config(config_path)
    model = Model(test_config)
    _, word_to_id = read_vocab(test_config['vocab_file'])
    categorys, cat_to_id = read_category()


    with True:
        line = input("请输入测试句子:")
        line = clean_str(line)
        x_input = [word_to_id[x] for x in line if x in word_to_id]
        predict = model.evaluate(x_input)
        print(categorys[predict])

def main(_):

    if FLAGS.is_train:
        if FLAGS.is_clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()

if __name__ == '__main__':
    tf.app.run(main)

