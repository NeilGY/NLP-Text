import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


class Model(object):
    #seq_len 是每个句子的长度(已词为单位,比如:I like you  3)
    #embedding_size 每个单词的词向量维度
    #vocab_size  所有句子对应的字典大小
    # num_class 分类维度
    #num_filter 卷积和数量
    #filter_sizes 不同卷积和大小的集合(比如shapeL:2*120;3*120)
    # l2_reg_lambda l2惩罚比例
    #inputs 输入数据的shape:数据量*每个句子分词后的维度
    #outputs 输出数据的shape:数据量*类别维度
    def __init__(self,config):
        self.seq_len = config['seq_len']
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.num_class = config['num_classes']
        self.num_filter = config['num_filters']
        self.filter_sizes = config['filter_sizes']

        self.learn_rate = config['lr']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.l2_loss = tf.constant(0.0)
        self.model_type = config['model_type']
        self.lstm_dim = config['lstm_dim']

        # self.dropout_keep = config['dropout_keep_prob']
        self.dropout_keep = tf.Variable(config['dropout_keep_prob'], name="global_step", trainable=False, dtype=tf.float32)
        self.inputs = tf.placeholder(tf.int32,shape=[None,self.seq_len])
        self.outputs = tf.placeholder(tf.float32,shape=[None,self.num_class])
        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        self.initializer = initializers.xavier_initializer()

        #TODO 根据模型类型增加
        if self.model_type == 'cnn':
            self.network = self.cnn_network()
        else:
            self.network = self.rnn_network()

        self.predict = tf.argmax(tf.nn.softmax(self.network), 1)
        with tf.variable_scope("loss"):#损失函数计算层
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.network,labels=self.outputs)) + self.l2_reg_lambda * self.l2_loss
        with tf.variable_scope("optimizer"):#优化损失层
            self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,global_step=self.global_step)
        with tf.variable_scope("Accuracy"):
            # softmax获得每个特征的占比(每条数据对应的特征数占比总和为1)
            # argmax获得最大占比的下标位置

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict,tf.argmax(self.outputs,1)),dtype=tf.float32))


    def cnn_network(self):
        with tf.device('/cpu:0'), tf.variable_scope("embedding_layer"):
            #例:embedding_lookup:[10*100],inputs:[2*5],则embedding_chars:
            #1.生成一个词向量字典:大小为[vocab_size,embedding_size]
            self.embedding_lookup = tf.get_variable("char_embedding",shape=[self.vocab_size, self.embedding_size],initializer=self.initializer)
            #2.查表 将inputs数据转换为三维的词向量数据 inputs+embedding_lookup[-1]
            self.embedding_chars = tf.nn.embedding_lookup(self.embedding_lookup,self.inputs)
            #卷积网络中是四维的 需要扩展一个维度
            embedding_chars_extend = tf.expand_dims(self.embedding_chars,-1)
        with tf.variable_scope("convolution_layer"):#卷积层：为不同大小的卷积核卷积,会生成多个feature map，并将其合并
            convs = []
            for i,filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("conv_{}".format(filter_size)):
                    #构建w,b
                    W = tf.get_variable("W".format(filter_size),shape=[filter_size,self.embedding_size,1,self.num_filter],initializer=self.initializer)
                    # B = tf.get_variable("B".format(filter_size),shape=[self.num_filter])
                    B = tf.get_variable("B".format(filter_size),shape=[self.num_filter],initializer=self.initializer)
                    #卷积
                    conv = tf.nn.conv2d(embedding_chars_extend,W,strides=[1,1,1,1],padding='VALID')
                    #激励
                    conv = tf.nn.relu6(tf.nn.bias_add(conv,B))
                    #池化层:整个句子做池化.公式:(seq_len-filter_size+1)/strudes
                    conv = tf.nn.max_pool(conv, ksize=[1, self.seq_len - filter_size+1,1,1],strides=[1,1,1,1],padding='VALID')
                    convs.append(conv)

            #对三个不同类型的卷积核合并,每种类型的卷积和数量是num_filter,
            num_filter_total = self.num_filter * len(self.filter_sizes)
            network = tf.concat(convs,3)
            network = tf.reshape(network,[-1,num_filter_total])

        with tf.variable_scope("softMax_layer"):#全连接的softMax层 输出每个类别的概率
            network = tf.nn.dropout(network,self.dropout_keep)
            W = tf.get_variable("softmax_w",shape=[num_filter_total,self.num_class],initializer=self.initializer)
            B = tf.get_variable("softmax_b",shape=[self.num_class],initializer=self.initializer)
            network = tf.nn.xw_plus_b(network,W,B)
            #增加l2损失函数
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(B)

        return  network

    def rnn_network(self):
        with tf.device('/cpu:0'), tf.variable_scope("embedding_layer"):
            #例:embedding_lookup:[10*100],inputs:[2*5],则embedding_chars:
            #1.生成一个词向量字典:大小为[vocab_size,embedding_size]
            self.embedding_lookup = tf.get_variable("char_embedding",shape=[self.vocab_size, self.embedding_size],initializer=self.initializer)
            #2.查表 将inputs数据转换为三维的词向量数据 inputs+embedding_lookup[-1]
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_lookup,self.inputs)

        with tf.variable_scope("rnn"):
            # 多层rnn网络
            def get_cells():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob = self.dropout_keep)
                gru_cell = tf.contrib.rnn.GRUCell(self.lstm_dim)
                gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell,output_keep_prob = self.dropout_keep)
                cells = [lstm_cell,gru_cell]
                return cells

            rnn_cell = tf.contrib.rnn.MultiRNNCell(get_cells(), state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedding_inputs, dtype=tf.float32)
            network = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
        with tf.variable_scope("rnn_softMax_layer"):#全连接的softMax层 输出每个类别的概率
            network = tf.nn.dropout(network,self.dropout_keep)
            W = tf.get_variable("W",shape=[self.lstm_dim,self.num_class],initializer=self.initializer)
            B = tf.get_variable("B",shape=[self.num_class],initializer=self.initializer)
            network = tf.nn.xw_plus_b(network,W,B)
        return network


    def run_step(self,sess,x_inputs,y_outputs):
        feed_dict = self.create_feed_dict(x_inputs,y_outputs)
        step,acc,los,_ = sess.run([self.global_step,self.accuracy,self.loss,self.train_op],feed_dict=feed_dict)
        return step,acc,los

    def text_step(self,sess,x_inputs,y_outputs):
        feed_dict = self.create_feed_dict(x_inputs,y_outputs)
        acc,los = sess.run([self.accuracy,self.loss],feed_dict=feed_dict)
        return acc,los

    def evaluate(self,sess,x_inputs):
        result = sess.run([self.predict],feed_dict=self.create_val_feed_dict(x_inputs))
        return result

    def create_feed_dict(self,x_inputs,y_outputs):
        feed_dict = {
            self.inputs:x_inputs,
            self.outputs:y_outputs
        }
        return feed_dict

    def create_val_feed_dict(self,x_inputs):
        feed_dict = {
            self.inputs:x_inputs,
            self.dropout_keep:1.0
        }
        return feed_dict