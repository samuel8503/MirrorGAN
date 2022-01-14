from __future__ import print_function

from miscc.config import cfg, cfg_from_file, cfg_clear
from datasets import TextDataset, Vocabulary
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
import traceback
import socket
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo, sentences):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    data_dic = {}
    # a list of indices for a sentence
    captions = []
    cap_lens = []
    sentences = [sentences, "a a a"]
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue

        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    key = "network_ipc"
    data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


def run(cfg_file, request):
    torch.cuda.empty_cache()
    cfg_clear()
    if cfg_file is not None:
        cfg_from_file(cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            try:
                gen_example(dataset.wordtoix, algo, request)  # generate images for customized captions
                result = {'success': True, 'msg': "Success!"}
            except:
                result = {'success': False, 'msg': traceback.format_exc()}
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
    return result

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = ''
    port = 22
    s.bind((host, port))
    while True:
    #for i in range(2):
        print("=" * 50)
        try:
            print("Listening...")
            s.listen()
            conn, addr = s.accept()
            print("Connected: {0}:{1}".format(addr[0], addr[1]))
            request = conn.recv(4096)
            request = pickle.loads(request)
            if request['dataset'] == 'bird':
                cfg_path = 'cfg/eval_bird.yml'
            else:
                cfg_path = 'cfg/eval_coco.yml'
            result = run(cfg_path, request['str'])
            print(request)
            #result = run('cfg/eval_bird.yml', 'this bird is red with white and has a very short beak')
            pprint.pprint(result)
            conn.sendall(pickle.dumps(result))
            conn.close()
        except KeyboardInterrupt:
            s.close()
            sys.exit(0)
        except:
            print(traceback.format_exc())
        print("=" * 50, end='\n\n')
