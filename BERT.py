# BERTの処理をする関数を作成する
import re
import os
import random
import collections
import time
import difflib
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import transformers
import japanize_matplotlib
import mojimoji
from transformers import BertForTokenClassification
import pickle
from transformers import logging



class tokenizer_for_wardlabeling(transformers.BertJapaneseTokenizer): 
    # 関数2-4-1-1:入力テキストのインデックス情報を保持しつつトークン分割を行う関数
    def position_tokenize(self, input_text, mecab_tokenizer):
        # 入力テキストをサブワードレベルでトークンに分割
        subword_tokens = self.tokenize(input_text)
        # 入力テキストをMeCabで形態素に分割
        morphologicals = mecab_tokenizer.tokenize(input_text)
        '''
        分割時の辞書は両トークナイザーとも同じものを使用
        self.tokenize：トークン（サブワード）に分割
        mecab_tokenizer.tokenize：形態素に分割（サブワードまでは分割されない）(メリット：未知語が無い)
        例）
            self.tokenize('カップラーメンを食べたら痒い')
                →　[カップ, ##ラー, ##メン, を, 食べ, たら, [UNK]]
            mecab_tokenizer.tokenize('カップラーメンを食べたら痒い')
                →　[カップラーメン, を,　食べ, たら，痒い] 
        '''
        # 結果を格納するリスト
        tokenize_token_dict_lis = []
        # 最初のトークンを識別するflag
        begin_token_flag = True
        # 現在地
        now_position = 0
        # 取り出したトークンに対応した形態素の位置情報を記録する変数
        morphological_idx = 0
        '''
        以下では，分割されたトークンに対して，もともとにテキストにおけるインデックスを付与
        する処理-----------------------------------------------------------------------
        '''
        
        for token in subword_tokens:
            # 未知語だった場合
            if token == '[UNK]':
                # 未知語リストから未知語を取り出す
                ori_token = morphologicals[morphological_idx]
                # 対応した形態素のインデックスを進める
                morphological_idx += 1
            # サブトークンであった場合
            elif token.startswith('##'):
                # サブトークンの「#」を削除
                ori_token = token.replace('#', '')
            else:
                # 普通のトークンの場合（形態素と同じ）
                ori_token = token
                # 対応した形態素のインデックスを進める
                morphological_idx += 1
            # トークンの文字数を確認
            ori_token_len = len(ori_token)
            # 開始位置を取得
            start = now_position
            # 終わり位置を取得
            finish = now_position + ori_token_len
            # それぞれの情報をdict型でまとめて返す
            token_dict = {
                '1_sub_word_token': token,
                '2_original_token': ori_token,
                '3_start_position': start,
                '4_finish_position': finish
            }
            tokenize_token_dict_lis.append(token_dict)
            # 現在地を更新
            now_position = finish
        return tokenize_token_dict_lis

# dictのkeyとvalueを入れ替える関数
def get_swap_dict(d):
    return {v: k for k, v in d.items()}

# 任意のテキストをBERTへの入力形式に変換する関数
def text2bertinput(
        input_text=None, 
        tokenizer=None,
        mecab_tokenizer=None,
        max_seq_len=None):
    #入力テキスト内の半角を全角に変換する
    input_text = mojimoji.han_to_zen(input_text)
    #特殊トークンの付与＆サブワード分割
    tokenize_txt = tokenizer.tokenize('[CLS]' + input_text + '[SEP]')
    #各id列の作成
    token_ids = tokenizer.convert_tokens_to_ids(tokenize_txt)
    mask_ids = [1] * len(token_ids)
    segment_ids = [0] * len(token_ids)
    #入力テキストの長さがモデルが指定する最大入力長になるまで[PAD]で水増し
    while len(token_ids) < max_seq_len:
        #[PAD]トークンの付与
        token_ids.append(0)
        mask_ids.append(0)
        segment_ids.append(0)
    #入力テキストが既に最大入力長を超えている場合、途中で切る。
    token_ids = token_ids[:max_seq_len]
    mask_ids = mask_ids[:max_seq_len]
    segment_ids = segment_ids[:max_seq_len]
    #各idの長さが適切かどうかの確認
    assert len(token_ids) == max_seq_len
    assert len(mask_ids) == max_seq_len
    assert len(segment_ids) == max_seq_len
    #各id列をtorch.tensorへ
    tensor_token_id = torch.tensor([token_ids], dtype=torch.long)
    tensor_mask_id = torch.tensor([mask_ids], dtype=torch.long)
    tensor_segment_id = torch.tensor([segment_ids], dtype=torch.long)
    return  {'token_ids': tensor_token_id,
             'mask_ids': tensor_mask_id,
             'segment_ids': tensor_segment_id}

# 関数7-3-2，入力文に対してラベル推定を行う関数
def label_prediction(model=None,
                     tokenizer=None,
                     mecab_tokenizer=None,
                     label2id_dic=None,
                     max_seq_len=None,
                     input_txt=None):
    '''
    分類ラベルに対応したidを戻すためのdictを作る----------------------------------
    '''
    label2id_dic = label2id_dic
    id2label_dic = get_swap_dict(label2id_dic)
    '''
    前準備----------------------------------------------------------------------
    '''
    # GPUの定義
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    # 今回はCPUなのでこっちがうごく(らしい)
    # 多分そう
    else:
        device = torch.device("cpu")
    # モデルをGPUへ
    model.to(device)
    torch.backends.cudnn.benchmark = True
    # モデルを検証モードへ
    model.eval()
    # BERTに入力した元々の文章を格納するリスト
    input_sentences = []
    # 正解したデータの数を数える
    epoch_corrects = 0
    # 経過したミニバッチを記録
    batch_processed_num = 0
    # 入力トークンの数
    num_all_token = 0
    '''
    入力テキストに対する処理----------------------------------------------------
    '''
    # テキストをBERTへ入力するためのid列に変換
    bert_ids = text2bertinput(input_text=input_txt, 
                              tokenizer=tokenizer,
                              max_seq_len=max_seq_len)
    # トークンid
    input_token_ids = bert_ids['token_ids'].to(device)
    # マスクid
    input_mask_ids = bert_ids['mask_ids'].to(device)
    # セグメントid
    input_segment_ids = bert_ids['segment_ids'].to(device)
    '''
    各idをBERTに入力、ラベル推論------------------------------------------
    '''    
    # BERTモデルへの入力
    results = model(input_ids=input_token_ids,
                    attention_mask=input_mask_ids,
                    token_type_ids=input_segment_ids)
    # 予測ラベルの取得
    _, preds = torch.max(results[0], 2)
     # トークンid中の入力トークンに対応したidのインデックスを抽出（特殊トークン以外）
    input_token_ids = input_token_ids[0]  
    token_idx = torch.where((input_token_ids!=0) & (input_token_ids!=2) \
                            & (input_token_ids!=3))
    # 特殊トークン以外の予測結果を抽出
    preds = preds[0]
    preds = preds[token_idx]
    '''
    入力された各トークンで予測されたラベルを確認------------------------
    '''
    # 入力テキストをトークン化(もとのインデックスを保持)
    wakati = tokenizer.position_tokenize(input_txt, mecab_tokenizer)
    # インデックス記録用
    index = 0
    # 各語彙にラベルを付与
    for token_dict, pred_label_id in zip(wakati, preds):
        if token_dict['1_sub_word_token'] == '[PAD]':
            break
        elif token_dict['1_sub_word_token'].startswith("##"):
            # サブトークンには直前の予測ラベルを適用
            token_dict['5_word_label'] = wakati[index-1]['5_word_label']
        else:
            # 予測されたラベルをidからラベルに変換
            pred_label = id2label_dic[pred_label_id.item()]
            # 結果をdictに追加
            token_dict['5_word_label'] = pred_label
        # インデックスを進める
        index += 1
    # 結果を返す
    return wakati

# 保存したチェックポイントを読み込む時に必要な関数
def model_load_checkpoint(model, load_path):
    """
        Args:
            model: 使用するBERTモデル（元となる事前学習済みモデル）
            load_path: 読み込むチェックポイントのpath
    """
    #checkpointの読み込み
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    
# word_labelsを受けとる関数
def get_bert_com(comment, 
model, 
tokenizer, 
mecab_tokenizer, 
label2id_dic):
    logging.set_verbosity_warning()
    labeling_result = label_prediction(
                        model=model,
                        tokenizer=tokenizer,
                        mecab_tokenizer=mecab_tokenizer,
                        label2id_dic=label2id_dic,
                        max_seq_len=51,
                        input_txt=comment)
    if labeling_result is not None:
        for word in labeling_result:
            if type(word) is dict:
                if word["5_word_label"] is not None:
                    if "動画に関するコメント" in word["5_word_label"]:
                        return comment
                    else:
                        return None


