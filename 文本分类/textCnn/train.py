# -*- coding:utf-8 -*-

import tensorflow as tf
from data_utils import load_data,char_mapping,shuffledData,getDivideData,iter_data,make_path,clean,get_logger,load_config,save_config,get_config
from model import Model
from collections import OrderedDict
import os


#数据的路径
tf.flags.DEFINE_string("positive_data_file", os.path.join('data','rt-polarity.pos'), "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", os.path.join('data','rt-polarity.neg'), "Data source for the negative data.")
tf.flags.DEFINE_string("vec_file", os.path.join('data/vec.txt'), "vocabe_file for data")
tf.flags.DEFINE_string("log_path", 'log', "")
tf.flags.DEFINE_string("config_path", 'config', "")
#模型参数
tf.flags.DEFINE_integer("embedding_size", 120, "Dimensionality of character embedding (default: 120)")
#指定三个四个或者五个单词卷积
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#每次卷积得到128个特征图
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
#dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#学习率
tf.flags.DEFINE_float("lr", 0.5, "learning rate (default:0.5)")
#L2的惩罚项
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")

# Training parameters  训练参数

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
#评估
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
#保存
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("checkpoints_path", 'result', "the path of checkpoints")

tf.flags.DEFINE_boolean("is_train", True, "train model")
tf.flags.DEFINE_boolean("is_clean", True, "clean train model")

FLAGS = tf.app.flags.FLAGS


# config for the model
def config_model(seq_len,vocab_size,num_classes,filter_sizes):
    config = OrderedDict()
    config["seq_len"] = seq_len
    config["vocab_size"] = vocab_size
    config["num_classes"] = num_classes
    config["vec_file"] = FLAGS.vec_file
    config["log_path"] = FLAGS.log_path
    config["embedding_size"] = FLAGS.embedding_size
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
    config["positive_data_file"] = FLAGS.positive_data_file
    config["negative_data_file"] = FLAGS.negative_data_file

    return config



def train():

    x_data,y_data = load_data(FLAGS.positive_data_file,FLAGS.positive_data_file)
    # seq_max:每行的最大维度。将数据中的每行按单词切割,取每行单词个数最多的为最终维度（每行的数据最终维度是一样的）
    seq_len = max([len(line.split()) for line in x_data])
    x_data,voca_processor = char_mapping(seq_len,x_data,FLAGS.vec_file)
    vocab_size = len(voca_processor.vocabulary_)
    num_classes = y_data.shape[-1]
    filter_sizes = [int(i) for i in FLAGS.filter_sizes.split(',')]
    #将数据划分为训练集和测试集
    x_train_data,y_train_data,x_test_data,y_test_data = getDivideData(x_data,y_data)

    # 生成文件的文件夹等
    make_path(FLAGS)
    logger = get_logger(os.path.join(FLAGS.log_path, 'train.log'))
    # 生成环境配置文件
    config_path = os.path.join(FLAGS.config_path, 'config')
    if not os.path.isfile(config_path):
        train_config = config_model(seq_len, vocab_size, num_classes,filter_sizes)
        save_config(train_config, config_path)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    with tf.Session(config=config) as sess:

        #创建模型
        model = Model(seq_len,FLAGS.embedding_size,vocab_size,num_classes,FLAGS.num_filters,filter_sizes,FLAGS.dropout_keep_prob,FLAGS.lr,FLAGS.l2_reg_lambda)
        sess.run(tf.global_variables_initializer())
        len_data = len(y_train_data)
        # model.network()
        #获取数据训练
        for i in range(FLAGS.num_epochs):
            for x_input,y_output in iter_data(x_train_data,y_train_data,FLAGS.batch_size):
                step, acc, loss = model.run_step(sess,x_input,y_output)

                #迭代100次评估一次模型
                if(step % FLAGS.evaluate_every == 0):
                    logger.info("train: iterator{}: step:{}/{} acc:{} loss:{} ".format(i+1,step%len_data,len_data,acc,loss ))
                    text_acc, text_los = model.text_step(sess,x_test_data,y_test_data)
                    logger.info("test: acc:{} loss:{} ".format(acc,loss ))
                    #保存模型
                    if acc>0.5 and text_acc > 0.5:
                        checkpoint_path = os.path.join(FLAGS.checkpoints_path,'checkpoints')
                        saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.max_to_keep)
                        saver.save(sess,checkpoint_path,global_step=step)


def evaluate_line():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(FLAGS.checkpoints_path, 'checkpoints')
    saver.restore(sess=session, save_path=checkpoint_path)  # 读取保存的模型
    config_path = os.path.join(FLAGS.config_path, 'config')
    train_config = load_config(config_path)
    model = Model(train_config['seq_len'], FLAGS.embedding_size, train_config['vocab_size'], train_config['num_classes'], FLAGS.num_filters, FLAGS.filter_sizes,
                  FLAGS.dropout_keep_prob, FLAGS.lr, FLAGS.l2_reg_lambda)
    with True:
        line = input("请输入测试句子:")
        result = model.evaluate(line)
        print(result)

def main(_):
    if FLAGS.is_train:
        if FLAGS.is_clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()

if __name__ == '__main__':
    tf.app.run(main)

