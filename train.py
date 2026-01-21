from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import shutil
import hashlib
from sys import platform
import data_utils
from data_utils import *
import argparse
import copy
import collections
from gensim.models import KeyedVectors
from model import GraphTransformer
import json

from tensorflow.python import pywrap_tensorflow
FLAGS = None

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="model/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="0", help="which gpu to use")

    parser.add_argument("--train_data", type=str, default="train", help="Training data path")
    parser.add_argument("--valid_data", type=str, default="valid", help="Valid data path")
    parser.add_argument("--test_data", type=str, default="test", help="Test data path")

    parser.add_argument("--from_vocab", type=str, default="data/cvocab", help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="data/evocab", help="to vocab path")
    parser.add_argument("--label_vocab", type=str, default="data/evocab", help="label vocab path")
    parser.add_argument("--output_dir", type=str, default="output/")

    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on size")
    parser.add_argument("--from_vocab_size", type=int, default=16500, help="source vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=16500, help="target vocabulary size")
    parser.add_argument("--edge_vocab_size", type=int, default=150, help="edge label vocabulary size")
    parser.add_argument("--enc_layers", type=int, default=6, help="Reduced layers for small dataset")
    parser.add_argument("--dec_layers", type=int, default=4, help="Number of layers in the decoder")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="Size of each model layer")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for small dataset")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--max_src_len", type=int, default=100, help="Max length src")
    parser.add_argument("--max_tgt_len", type=int, default=100, help="Max length tgt")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--use_copy", type=int, default=True, help="Whether use copy mechanism")
    parser.add_argument("--use_depth", type=int, default=False, help="Whether use depth embedding")
    parser.add_argument("--use_charlstm", type=int, default=False, help="Whether use character embedding")
    parser.add_argument("--input_keep_prob", type=float, default=1.0, help="Dropout input keep prob")
    parser.add_argument("--output_keep_prob", type=float, default=0.9, help="Dropout output keep prob")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.5)

def create_hparams(flags):
    return tf.contrib.training.HParams(
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        output_dir=flags.output_dir,
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        edge_vocab_size=flags.edge_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,
        emb_dim=flags.emb_dim,
        max_train_data_size=flags.max_train_data_size,
        train_data=flags.train_data,
        valid_data=flags.valid_data,
        test_data=flags.test_data,
        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,
        label_vocab=flags.label_vocab,
        share_vocab=False,
        use_copy=flags.use_copy,
        use_depth=flags.use_depth,
        use_charlstm=flags.use_charlstm,
        input_keep_prob=flags.input_keep_prob,
        output_keep_prob=flags.output_keep_prob,
        dropout_rate=flags.dropout_rate,
        init_weight=0.1,
        num_units=flags.num_units,
        num_heads=flags.num_heads,
        enc_layers=flags.enc_layers,
        dec_layers=flags.dec_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        max_src_len=flags.max_src_len,
        max_tgt_len=flags.max_tgt_len,
        max_seq_length=42,
        epoch_num=flags.epoch_num,
        epoch_step=0,
        lambda1=flags.lambda1,
        lambda2=flags.lambda2
    )

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model"))): pass
class EvalModel(collections.namedtuple("EvalModel", ("graph", "model"))): pass
class InferModel(collections.namedtuple("InferModel", ("graph", "model"))): pass

def create_model(hparams, model, length=22):
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)
    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(graph=infer_graph, model=infer_model)

def read_data_graph(src_path, edge_path, ref_path, wvocab, evocab, cvocab, hparams):
    data_set = []
    unks = []
    skipped_count = 0
    
    with tf.gfile.GFile(src_path, mode="r") as src_file, \
         tf.gfile.GFile(edge_path, mode="r") as edge_file, \
         tf.gfile.GFile(ref_path, mode="r") as ref_file:
        
        for src, edges, ref in zip(src_file, edge_file, ref_file):
            src_clean = src.lower().strip()
            edges_clean = edges.strip()
            ref_clean = ref.strip()
            
            if not src_clean or not edges_clean or not ref_clean:
                skipped_count += 1
                continue
            
            src_seq = src_clean.split(" ")
            tgt = ref_clean.lower().split(" ")
            try:
                graph = json.loads(edges_clean)
            except:
                skipped_count += 1
                continue
            
            src_ids = []
            tgt_ids = []
            char_ids = []
            unk = []
            edges_list = []
            depth = []
            
            # Process nodes
            for w in src_seq:
                if not w: continue
                # Char IDs (Dung mapping cvocab hoac mac dinh 76)
                char_id = [cvocab.get(c, 76) for c in w]
                char_ids.append(char_id)
                # Word IDs
                src_ids.append(wvocab.get(w, data_utils.UNK_ID))
                unk.append(w)
                depth.append([])
            
            node_count = len(src_ids)
            if depth: depth[0].append(0)

            # Process target
            for w in tgt:
                tgt_ids.append(wvocab.get(w, data_utils.UNK_ID))

            # Process edges
            dicc = {}
            for l_node in graph:
                id1 = int(l_node)
                if id1 >= node_count: continue
                
                for pair in graph[l_node]:
                    edge_label, id2 = pair[0], int(pair[1])
                    if id2 >= node_count: continue
                    
                    if edge_label == ":mode" or edge_label == ":polarity":
                        if id2 not in dicc:
                            dicc[id2] = 1
                        else:
                            w_val = src_seq[id2]
                            src_ids.append(wvocab.get(w_val, data_utils.UNK_ID))
                            char_ids.append([cvocab.get(c, 76) for c in w_val])
                            unk.append(w_val)
                            tmp = depth[id1].copy()
                            tmp.append(evocab.get(edge_label, data_utils.UNK_ID))
                            depth.append(tmp)
                            id2 = len(src_ids) - 1

                    edge_id = evocab.get(edge_label, data_utils.UNK_ID)
                    if id2 < len(depth) and not depth[id2]:
                        tmp = depth[id1].copy()
                        tmp.append(edge_id)
                        depth[id2] = tmp

                    edges_list.append([edge_id, id1, id2])
            
            if len(src_ids) < hparams.max_src_len and len(tgt_ids) < hparams.max_tgt_len:
                data_set.append([src_ids, tgt_ids, edges_list, char_ids, depth, ref_clean])
                unks.append(unk)
                    
    print("load data finish: {0} lines. Skipped: {1} lines.".format(len(data_set), skipped_count))
    return data_set, unks

def train(hparams):
    # Tu dong build vocab neu rong (Tuong thich Python 3.5 format)
    wvocab, _ = data_utils.initialize_vocabulary(hparams.from_vocab, os.path.join(hparams.data_dir, "train.src"))
    evocab, _ = data_utils.initialize_vocabulary(hparams.label_vocab, os.path.join(hparams.data_dir, "train.tgt"))
    cvocab, _ = data_utils.initialize_vocabulary("data/cvocab", os.path.join(hparams.data_dir, "train.src"))

    embeddings = init_embedding(hparams, wvocab)
    hparams.add_hparam(name="embeddings", value=embeddings)
    
    train_model, eval_model, infer_model = create_model(hparams, GraphTransformer)
    config = get_config_proto()
    
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)
    
    train_data, _ = read_data_graph(os.path.join(hparams.data_dir, "train.src"), 
                                    os.path.join(hparams.data_dir, "train.edge"),
                                    os.path.join(hparams.data_dir, "train.tgt"),
                                    wvocab, evocab, cvocab, hparams)
    valid_data, _ = read_data_graph(os.path.join(hparams.data_dir, "valid.src"), 
                                    os.path.join(hparams.data_dir, "valid.edge"),
                                    os.path.join(hparams.data_dir, "valid.tgt"),
                                    wvocab, evocab, cvocab, hparams)

    ckpt = tf.train.get_checkpoint_state(hparams.train_dir)
    ckpt_path = os.path.join(hparams.train_dir, "ckpt")
    
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())
            global_step = 0

    epoch_step = int((len(train_data) - 1) / hparams.batch_size) + 1
    now_step = 0
    total_loss, total_time = 0.0, 0.0
    
    print("Bắt đầu huấn luyện...")
    epoch_count = 0
    while global_step <= 150000:
        start_time = time.time()
        step_loss, global_step, predict_count = train_model.model.train_step(train_sess, train_data, no_random=True, id=now_step * hparams.batch_size)
        
        now_step += 1
        # Khi kết thúc 1 Epoch
        if now_step >= epoch_step:
            epoch_count += 1
            now_step = 0
            random.shuffle(train_data)
            # In ra tiến độ sau mỗi vòng lặp để bạn biết nó đang chạy đến đâu
            print("--- Kết thúc Epoch {0}/{1} | Global Step: {2} ---".format(epoch_count, hparams.epoch_num, global_step))
        
        total_loss += step_loss
        total_time += (time.time() - start_time)
        
        # In chi tiết Loss sau mỗi 100 bước
        if global_step % 10 == 0:
            avg_loss = total_loss / 100
            avg_time = total_time / 100
            print(">> [Báo cáo] Step {0} | Loss trung bình: {1:.4f} | Time: {2:.2f}s".format(global_step, avg_loss, avg_time))
            total_loss, total_time = 0.0, 0.0
            
        if now_step == 0:
            train_model.model.saver.save(train_sess, ckpt_path, global_step=global_step)
            print(">> Đã lưu mô hình (checkpoint) tại step {0}".format(global_step))

def init_embedding(hparams, vocab):
    emb = np.random.uniform(-0.05, 0.05, (len(vocab), hparams.emb_dim)).astype(np.float32)
    if os.path.exists("data/amr_vector.txt"):
        try:
            word_vectors = KeyedVectors.load_word2vec_format("data/amr_vector.txt")
            for word, idx in vocab.items():
                if word in word_vectors:
                    emb[idx] = word_vectors[word]
        except:
            print("Load amr_vector failed, using random initialization.")
    return emb

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run(main=lambda _: train(create_hparams(FLAGS)))