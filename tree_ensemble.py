"""
author:humingyue
date:2018-9-28
description:ensemble multiple cnn models
"""
import os
import time
import numpy as np
import tensorflow as tf
from utils import *
from PIL import Image
from tensorflow.python import pywrap_tensorflow

classnames = ['AmericanBeech', 'AmericanSycamore', 'BlackWalnut', 'EasternRedCedar', 'Ginkgo', 'RedMaple',
                  'SouthernMagnolia', 'TulipPoplar', 'WhiteOak', 'WhitePine']

def load_alexnet(sess,image_path,checkpoint_path):
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    conv1_weights = reader.get_tensor('conv1/weights')
    conv1_biases = reader.get_tensor('conv1/biases')
    conv2_weights = reader.get_tensor('conv2/weights')
    conv2_biases = reader.get_tensor('conv2/biases')
    conv3_weights = reader.get_tensor('conv3/weights')
    conv3_biases = reader.get_tensor('conv3/biases')
    conv4_weights = reader.get_tensor('conv4/weights')
    conv4_biases = reader.get_tensor('conv4/biases')
    conv5_weights = reader.get_tensor('conv5/weights')
    conv5_biases = reader.get_tensor('conv5/biases')
    fc6_weights = reader.get_tensor('fc6/weights')
    fc6_biases = reader.get_tensor('fc6/biases')
    fc7_weights = reader.get_tensor('fc7/weights')
    fc7_biases = reader.get_tensor('fc7/biases')
    fc8_weights = reader.get_tensor('fc8/weights')
    fc8_biases = reader.get_tensor('fc8/biases')
    conv1 = conv2d(x, conv1_weights, conv1_biases, 4, 4, padding="VALID", groups=1)
    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
    conv2 = conv2d(pool1, conv2_weights, conv2_biases, 1, 1, groups=2)
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
    conv3 = conv2d(pool2, conv3_weights, conv3_biases, 1, 1, padding="SAME", groups=1)
    conv4 = conv2d(conv3, conv4_weights, conv4_biases, 1, 1, padding="SAME", groups=2)
    conv5 = conv2d(conv4, conv5_weights, conv5_biases, 1, 1, padding="SAME", groups=2)
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, fc6_weights, fc6_biases)
    dropout6 = dropout(fc6, 1)
    fc7 = fc(dropout6, fc7_weights, fc7_biases)
    dropout7 = dropout(fc7, 1)
    fc8 = fc(dropout7, fc8_weights, fc8_biases, relu=False)
    softmax = tf.nn.softmax(fc8)
    imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    im = Image.open(image_path).resize((227, 227))
    im = np.array(im, dtype=np.float32)
    im-=imagenet_mean
    im = np.reshape(im, [1, 227, 227, 3])
    sess.run(tf.global_variables_initializer())
    probs = sess.run(softmax, feed_dict={x: im})
    top1 = probs.argsort().reshape(10)[::-1][0]
    # print('id:[%d] name:[%s] (score = %.5f)' % (top1,classnames[top1], probs[0,top1]))
    # print(probs.reshape(10))
    return (top1,probs.reshape(10))

def load_vggnet_16(sess,image_path,model_file,label_file):
    create_graph(model_file)
    softmax_tensor = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = open(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)
    predictions = tf.nn.softmax(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(label_file)
    top_names = []
    top_1 = predictions.eval().argsort()[::-1][0]
    human_string = node_lookup.id_to_string(top_1)
    top_names.append(human_string)
    score = predictions.eval()[top_1]
    # print('id:[%d] name:[%s] (score = %.5f)' % (top_1, human_string, score))
    # print(predictions.eval())
    return (top_1,predictions.eval())

def load_resnet_50(sess,image_path,model_file,label_file):
    create_graph(model_file)
    softmax_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/SpatialSqueeze:0')
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = open(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)
    predictions = tf.nn.softmax(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(label_file)
    top_names = []
    top_1 = predictions.eval().argsort()[::-1][0]
    human_string = node_lookup.id_to_string(top_1)
    top_names.append(human_string)
    score = predictions.eval()[top_1]
    # print('id:[%d] name:[%s] (score = %.5f)' % (top_1, human_string, score))
    # print(predictions.eval())
    return (top_1,predictions.eval())

def load_rt_inception_v3(sess,image_path,model_file,label_file):
    create_graph(model_file)
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = open(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)
    predictions = tf.nn.softmax(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(label_file)
    top_names = []
    top_1 = predictions.eval().argsort()[::-1][0]
    human_string = node_lookup.id_to_string(top_1)
    top_names.append(human_string)
    score = predictions.eval()[top_1]
    # print('id:[%d] name:[%s] (score = %.5f)' % (top_1, human_string, score))
    # print(predictions.eval())
    return (top_1,predictions.eval())

def load_inception_v3(sess,image_path,model_file,label_file):
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    image_data = open(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
    predictions = np.squeeze(predictions)  # 把结果转为1维数据
    predictions=np.array([predictions[i] for i in change_sort(label_file)])#改变顺序
    # 排序
    top_1 = predictions.argsort()[::-1][0]
    human_string = classnames[top_1]
    score = predictions[top_1]
    # print('id:[%d] name:[%s] (score = %.5f)' % (top_1,human_string, score))
    # print(predictions)
    return (top_1,predictions)

def get_key(dict,value):
    for k, v in dict.items():
        if v == value:
            return k

#原始数据迁移训练
def model_ensemble_1x(image_path):
   #alexnet保存模型
    alexnet_fc876_path ='E:/Humy/fineturn_alexnet/result/alex_04/checkpoints/model_epoch100.ckpt'

    #vggnet-16保存模型
    vgg_fc876_path = 'E:/Humy/pb/ft_vgg_16_fc876_1x_freeze.pb'
    vgg_fc876_label='E:/Humy/pb/1x_labels.txt'

    #inception-v3保存模型
    inception_path='E:/Humy/ExpModel/tree/00/output_graph.pb'
    inception_label='E:/Humy/ExpModel/tree/00/output_labels.txt'

    #resnet-50保存模型
    resnet_path='E:/Humy/pb/ft_resnet_50_1x_freeze.pb'
    resnet_label='E:/Humy/pb/1x_labels.txt'

    g1=tf.Graph()
    g2=tf.Graph()
    g3=tf.Graph()
    g4 = tf.Graph()

    with tf.Session(graph=g1) as sess1:
        (id1,alexnet) = load_alexnet(sess1,image_path,alexnet_fc876_path)
    with tf.Session(graph=g2) as sess2:
        (id2,vggnet_16) = load_vggnet_16(sess2,image_path, vgg_fc876_path, vgg_fc876_label)
    with tf.Session(graph=g3) as sess3:
        (id3,inception_v3) = load_inception_v3(sess3,image_path,inception_path,inception_label)
    with tf.Session(graph=g4) as sess4:
        (id4,resnet_50) = load_resnet_50(sess4,image_path,resnet_path,resnet_label)

    ensemble=np.array([(2/10)*alexnet,(3/10)*vggnet_16,(1/10)*inception_v3,(4/10)*resnet_50])
    average=np.sum(ensemble,axis=0)

    result_list = [id1, id2, id3, id4]
    set_result_list = set(result_list)
    result={}

    for i in set_result_list:
        result[i]=result_list.count(i)
    max=0

    for k,v in result.items():
        if v>max:
            max=v

    id=get_key(result,max)
    return (id,average)

#10倍数据迁移训练
def model_ensemble_10x(image_path):
    # alexnet保存模型
    alexnet_fc876_path_10x = 'E:/Humy/fineturn_alexnet/result_10x/alex_03/checkpoints/model_epoch100.ckpt'

    # vggnet-16保存模型
    vgg_fc876_path_10x = 'E:/Humy/pb/ft_vgg_16_fc876_10x_freeze.pb'
    vgg_fc876_label_10x = 'E:/Humy/pb/10x_labels.txt'

    # inception-v3保存模型
    inception_path_10x = 'E:/Humy/ExpModel/second/02/output_graph.pb'
    inception_label_10x = 'E:/Humy/ExpModel/second/02/output_labels.txt'

    # resnet保存模型
    resnet_path_10x = 'E:/Humy/pb/ft_resnet_50_10x_freeze.pb'
    resnet_label_10x = 'E:/Humy/pb/10x_labels.txt'

    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()
    g4 = tf.Graph()

    with tf.Session(graph=g1) as sess1:
        (id1,alexnet) = load_alexnet(sess1, image_path, alexnet_fc876_path_10x)
    with tf.Session(graph=g2) as sess2:
        (id2,vggnet_16) = load_vggnet_16(sess2, image_path, vgg_fc876_path_10x, vgg_fc876_label_10x)
    with tf.Session(graph=g3) as sess3:
        (id3,inception_v3) = load_inception_v3(sess3, image_path, inception_path_10x, inception_label_10x)
    with tf.Session(graph=g4) as sess4:
        (id4,resnet_50) = load_resnet_50(sess4, image_path, resnet_path_10x, resnet_label_10x)

    ensemble = np.array([(1/10)*alexnet, (4/10)*vggnet_16, (3/10)*inception_v3, (2/10)*resnet_50])
    average = np.sum(ensemble, axis=0)
    result_list = [id1, id2, id3, id4]
    set_result_list = set(result_list)
    result = {}

    for i in set_result_list:
        result[i] = result_list.count(i)
    max = 0

    for k, v in result.items():
        if v > max:
            max = v

    id = get_key(result, max)
    return (id, average)


#原始数据从头训练
def rt_model_ensemble_1x(image_path):
   #alexnet保存模型
    alexnet_path ='E:/Humy/fineturn_alexnet/result/restart/alex_01/checkpoints/model_epoch100.ckpt'

    #vggnet-16保存模型
    vgg_path = 'E:/Humy/pb/rt_vgg_16_1x_freeze.pb'
    vgg_label='E:/Humy/pb/1x_labels.txt'

    #inception-v3保存模型
    inception_path='E:/Humy/pb/rt_inception_v3_1x_freeze.pb'
    inception_label='E:/Humy/pb/1x_labels.txt'

    #resnet-50保存模型
    resnet_path='E:/Humy/pb/rt_resnet_50_1x_freeze.pb'
    resnet_label='E:/Humy/pb/1x_labels.txt'

    g1=tf.Graph()
    g2=tf.Graph()
    g3=tf.Graph()
    g4 = tf.Graph()

    with tf.Session(graph=g1) as sess1:
        (id1,alexnet) = load_alexnet(sess1,image_path,alexnet_path)
    with tf.Session(graph=g2) as sess2:
        (id2,vggnet_16) = load_vggnet_16(sess2,image_path, vgg_path, vgg_label)
    with tf.Session(graph=g3) as sess3:
        (id3,inception_v3) = load_rt_inception_v3(sess3,image_path,inception_path,inception_label)
    with tf.Session(graph=g4) as sess4:
        (id4,resnet_50) = load_resnet_50(sess4,image_path,resnet_path,resnet_label)

    ensemble=np.array([(1/10)*alexnet,(3/10)*vggnet_16,(4/10)*inception_v3,(2/10)*resnet_50])
    average=np.sum(ensemble,axis=0)

    result_list = [id1, id2, id3, id4]
    set_result_list = set(result_list)
    result={}

    for i in set_result_list:
        result[i]=result_list.count(i)
    max=0

    for k,v in result.items():
        if v>max:
            max=v

    id=get_key(result,max)
    return (id,average)


#10倍数据从头训练
def rt_model_ensemble_10x(image_path):
    # alexnet保存模型
    alexnet_path_10x = 'E:/Humy/fineturn_alexnet/result_10x/restart/alex_02/checkpoints/model_epoch100.ckpt'

    # vggnet-16保存模型
    vgg_path_10x = 'E:/Humy/pb/rt_vgg_16_10x_freeze.pb'
    vgg_label_10x = 'E:/Humy/pb/10x_labels.txt'

    # inception-v3保存模型
    inception_path_10x = 'E:/Humy/pb/rt_inception_v3_10x_freeze.pb'
    inception_label_10x = 'E:/Humy/pb/10x_labels.txt'

    # resnet保存模型
    resnet_path_10x = 'E:/Humy/pb/rt_resnet_50_10x_freeze.pb'
    resnet_label_10x = 'E:/Humy/pb/10x_labels.txt'

    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()
    g4 = tf.Graph()

    with tf.Session(graph=g1) as sess1:
        (id1,alexnet) = load_alexnet(sess1, image_path, alexnet_path_10x)
    with tf.Session(graph=g2) as sess2:
        (id2,vggnet_16) = load_vggnet_16(sess2, image_path, vgg_path_10x, vgg_label_10x)
    with tf.Session(graph=g3) as sess3:
        (id3,inception_v3) = load_rt_inception_v3(sess3, image_path, inception_path_10x, inception_label_10x)
    with tf.Session(graph=g4) as sess4:
        (id4,resnet_50) = load_resnet_50(sess4, image_path, resnet_path_10x, resnet_label_10x)

    ensemble = np.array([(1/10)*alexnet, (4/10)*vggnet_16, (3/10)*inception_v3, (2/10)*resnet_50])
    average = np.sum(ensemble, axis=0)
    result_list = [id1, id2, id3, id4]
    set_result_list = set(result_list)
    result = {}

    for i in set_result_list:
        result[i] = result_list.count(i)
    max = 0

    for k, v in result.items():
        if v > max:
            max = v

    id = get_key(result, max)
    return (id, average)

def evaluate(image_path):
    (id,average) = model_ensemble_1x(image_path)

    a = np.argmax(average, 0)
    print(id)
    print('最终结果为：id:[%d] name:[%s] (score = %.5f)' % (a, classnames[a], average[a]))


if __name__ == "__main__":
    INPUT_DATA = 'E:\\Humy\\Slim_exp\\data\\data_1x\\tree\\eval.txt'  # 原始数据集
    INPUT_DATA_10x = 'E:\\Humy\\Slim_exp\\data\\data_10x\\tree\\eval.txt'  # 10倍数据增强
    time1 = time.time()
    print("开始测试……")
    i = 0#总数
    j = 0#加权平均法验证正确数
    k = 0#投票法验证正确数
    with open(INPUT_DATA_10x, "r") as f:
        lines = f.readlines()
        for line in lines:
            i+=1
            path_label = line.rsplit(" ")
            path = path_label[0]
            label = path_label[1]
            (id,average) = model_ensemble_10x(path)
            print(id,average)
            a = np.argmax(average, 0)
            if a==int(label):
                j+=1
                print("第%d张加权平均法预测正确！"%i)
            else:
                print("第%d张加权平均法预测错误！" % i)

            if id==int(label):
                k += 1
                print("第%d张投票法预测正确！" % i)
            else:
                print("第%d张投票法预测错误！" % i)

    print("加权平均法准确率为:%s/%s=%s" % (j, i, j / i))
    print("投票法准确率为:%s/%s=%s" % (k, i, k / i))

    # image_path = "E:\\Humy\\pb\\test\\WhitePine.jpg"
    # image_path = "E:\\Humy\\Slim_exp\\data\\data_1x\\tree\\AmericanSycamore\\IMG_2f8d09b5-f980-4062-9d96-dab0186261639110.jpg"
    #     # for image in os.listdir(path):
    #     # image_path = os.path.join(path, image)
    #     # print(change_sort(inception_v3_label_file))
    # evaluate(image_path)
    time2 = time.time()
    print("cost:",time2-time1)
