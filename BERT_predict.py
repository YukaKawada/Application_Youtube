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
from torch.utils.data import DataLoader, \
  RandomSampler, SequentialSampler, TensorDataset
import pickle
# transformersライブラリをインポート
import transformers
import mojimoji

# 事前学習モデルのpath
pretrained_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'

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




