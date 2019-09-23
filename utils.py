import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""
    def __init__(self,
                 label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)
    def load(self, label_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human
        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def create_graph(model_file=None):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    if not model_file:
        model_file = model_file
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def show_ckpt(path):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key, "shape:", reader.get_tensor(key).shape)
    # chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name='', all_tensors=True)#can print weight by this

def fc(x, weights, biases,relu=True):
    act = tf.nn.xw_plus_b(x, weights, biases)
    if relu:
        relu = tf.nn.relu(act)
        return relu
    else:
        return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

#卷积层
def conv2d(x,W,b,stride_y,stride_x,padding="SAME",groups=1):
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    if groups==1:
        conv=convolve(x,W)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=W)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(axis=3, values=output_groups)
    bias = tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))
    relu = tf.nn.relu(bias)
    return relu

def change_sort(label_path):
    lines = tf.gfile.GFile(label_path).readlines()
    uid_to_human = {}
    for uid, line in enumerate(lines):
        # 去掉换行符
        line = line.strip('\n')
        uid_to_human[uid] = line
    def get_key(dict,value):
        for k, v in dict.items():
            if v == value:
                return k
        # return [k for k, v in dict.items() if v == value]
    right_tree_list=['americanbeech','americansycamore','blackwalnut','easternredcedar','ginkgo','redmaple',
                     'southernmagnolia','tulippoplar','whiteoak','whitepine']
    sorted_list = []
    for i in right_tree_list:
        sorted_list.append(get_key(uid_to_human,i))
    # print(uid_to_human)
    return sorted_list
