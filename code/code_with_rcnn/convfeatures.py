import tensorflow as tf
import numpy as np
import os
import sys
from  more_itertools import unique_everseen


batch_size = 1

img_path1 = "lorem"
img_path2 = "ipsum"
files = 1
#src = os.path.dirname(os.path.abspath(__file__))
#        print ("Directory exists")
 #   else:
 #       print("Directory does not exists")
        

with open('ConvNets/inception_v4.pb', 'rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def)
graph = tf.get_default_graph()

input_layer = graph.get_tensor_by_name("import/InputImage:0")
output_layer = graph.get_tensor_by_name(
    "import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")



def build_prepro_graph():
    input_file = tf.placeholder(dtype=tf.string, name="InputFile")
    image_file = tf.read_file(input_file)
    jpg = tf.image.decode_jpeg(image_file, channels=3)
    png = tf.image.decode_png(image_file, channels=3)
    output_jpg = tf.image.resize_images(jpg, [299, 299]) / 255.0
    output_jpg = tf.reshape(
        output_jpg, [
            1, 299, 299, 3], name="Preprocessed_JPG")
    output_png = tf.image.resize_images(png, [299, 299]) / 255.0
    output_png = tf.reshape(
        output_png, [
            1, 299, 299, 3], name="Preprocessed_PNG")
    return input_file, output_jpg, output_png


def load_image(sess, io, image):
    if image.split('.')[-1] == "png":
        return sess.run(io[2], feed_dict={io[0]: image})
    return sess.run(io[1], feed_dict={io[0]: image})


def load_next_batch(sess, io):
    for batch_idx in range(0, len(files), batch_size):
        batch1 = files[batch_idx:batch_idx + batch_size]
        batch1 = np.array(
            map(lambda x: load_image(sess, io, img_path1 + x), batch1))
        batch1 = batch1.reshape((batch_size, 299, 299, 3))
        yield batch1

def load_next_batch2(sess, io):
    for batch_idx in range(0, len(files), batch_size):
        batch2 = files[batch_idx:batch_idx + batch_size]
        batch2 = np.array(
            map(lambda x: load_image(sess, io, img_path1 + x), batch2))
        batch2 = batch2.reshape((batch_size, 299, 299, 3))
        yield batch2

def forward_pass(io):
    global output_layer
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_iter1 = load_next_batch(sess, io)
        batch_iter2 = load_next_batch2(sess, io)
        for i in xrange(n_batch):
            batch1 = batch_iter1.next()
            batch2 = batch_iter2.next()

            assert batch1.shape == (batch_size, 299, 299, 3)
            assert batch2.shape == (batch_size, 299, 299, 3)
            
            feed_dict1 = {input_layer: batch1}
            feed_dict2 = {input_layer: batch2}
            if i is 0:
                prob1 = sess.run(
                    output_layer, feed_dict=feed_dict1).reshape(
                    batch_size, 1536)

                prob2 = sess.run(
                    output_layer, feed_dict=feed_dict2).reshape(
                    batch_size, 1536)

                prob = np.concatenate((prob1, prob2), axis=1)

            else:
                prob1 = sess.run(
                    output_layer, feed_dict=feed_dict1).reshape(
                    batch_size, 1536)

                prob2 = sess.run(
                    output_layer, feed_dict=feed_dict2).reshape(
                    batch_size, 1536)

                prob3 = np.concatenate((prob1, prob2), axis=1)

                prob = np.append(
                    prob, prob3, axis=0)

            if i % 5 == 0:
                print "Progress:" + str(((i + 1) / float(n_batch) * 100)) + "%\n"
    print "Progress:" + str(((n_batch) / float(n_batch) * 100)) + "%\n"
    print
    print "Saving Features : ",str(sys.argv)[4],"\n"
#    np.save('Data/features', prob)
    np.save(str(sys.argv[4]), prob)


def get_features(sess, io, img, saveencoder=False):
    global output_layer
    output_layer = tf.reshape(output_layer, [1,1536], name="Output_Features")
    image = load_image(sess, io, img)
    feed_dict = {input_layer: image}
    prob = sess.run(output_layer, feed_dict=feed_dict)

    if saveencoder:
        tensors = [n.name for n in sess.graph.as_graph_def().node]
        with open("model/Encoder/Encoder_Tensors.txt", 'w') as f:
            for t in tensors:
                f.write(t + "\n")
        saver = tf.train.Saver()
        saver.save(sess, "model/Encoder/model.ckpt")
    return prob

if __name__ == "__main__":
    
    # global img_path1, img_path2, files

    print("Generating features for images")

    if len(sys.argv)!=5:
        print("The correct syntax is: python convfeatures.py 'image_folder_path 1' 'image_folder_path 2' 'captions_file' 'feature_save_path'")
        exit(0)

    img_path1 = str(sys.argv[1])
    img_path2 = str(sys.argv[2])
    cap_file = str(sys.argv[3])
    #img_path = os.path.join(src, img_path)

    try:
        if ( (not os.path.isdir(img_path1)) or (not os.path.isdir(img_path2)) ):
            exit(0)
                 
        # files = sorted(np.array(os.listdir(img_path1)))
        with open(cap_file, 'r') as f:
            data = f.readlines()
        files = [caps.split('\t')[0].split('#')[0] for caps in data]
        files = list(unique_everseen(files))

        n_batch = len(files) / batch_size
    except:
        pass


    
    print "#Images:", len(files)
    print "Extracting Features"
    io = build_prepro_graph()
    forward_pass(io)
    print "done"
