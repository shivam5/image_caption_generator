from caption_generator import *
from utils.data_util import generate_captions
from configuration import Configuration
import os
import argparse
import json
from  more_itertools import unique_everseen

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help="train|test|eval",
    choices=[
        "train",
        "test",
        "eval"],
    required=True)
parser.add_argument(
    "--resume",
    type=int,
    help="1=Yes|0=No",
    choices=[
        1,
        0],
    default=0)
parser.add_argument(
    "--saveencoder",
    type=int,
    help="1=Yes|0=No",
    choices=[
        1,
         0])
parser.add_argument(
    "--savedecoder",
    type=int,
    help="1=Yes|0=No",
    choices=[
        1,
         0])
parser.add_argument(
    "--image_path",
    type=str,
    help="Path to the Image for Generation of Captions")
parser.add_argument(
    "--validation_data",
    type=str,
    help="Path to the Test Data for evaluation")
args = parser.parse_args()
config = Configuration(vars(args))

if config.mode == "train":
    caption_file = 'Data/training.txt'
    feature_file = 'Data/training_features.npy'
    vocab, wtoidx, training_data = generate_captions(
        config.word_threshold, config.max_len, caption_file, feature_file)
    features, captions = training_data[:, 0], training_data[:, 1]
    features = np.array([feat.astype(float) for feat in features])
    data = (vocab.tolist(), wtoidx.tolist(), features, captions)
    model = Caption_Generator(config, data=data)
    loss, inp_dict = model.build_train_graph()
    model.train(loss, inp_dict)

elif config.mode == "test":
    if os.path.exists(args.image_path):
        model = Caption_Generator(config)
        model.decode(args.image_path)
    else:
        print "Please provide a valid image path.\n Usage:\n python main.py --mode test --image_path VALID_PATH"

elif config.mode == "eval":
    config.mode = "test"
    config.batch_decode = True
    print args.validation_data
    if os.path.exists(args.validation_data):
        features = np.load(args.validation_data)

        # filenames = sorted(np.array(os.listdir("Data/Test/")))
        cap_file = "Data/test.txt"
        with open(cap_file, 'r') as f:
            data = f.readlines()
        files = [caps.split('\t')[0].split('#')[0] for caps in data]
        filenames = list(unique_everseen(files))

        # with open("Data/image_info_test2014.json",'r') as f:
        #     data=json.load(f)

        #filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
        # filenames  = sorted([d["file_name"].split('.')[0] for d in data['images']])
        #captions = [caps.replace('\n', '').split('\t')[1] for caps in data]
        #features, captions = validation_data[:, 0], validation_data[:, 1]

        features = np.array([feat.astype(float) for feat in features])
        model = Caption_Generator(config)
        generated_captions = model.batch_decoder(filenames, features)
