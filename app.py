from flask import Flask, request, render_template, send_from_directory
from apiclient.discovery import build
from apiclient.errors import HttpError
from transformers import BertForTokenClassification
import mycode
import pickle
import BERT
import transformers


app = Flask(__name__,
            static_url_path='',
            static_folder='static')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['GET'])
def input_form():
    query = request.args.get('query', '')

    # クエリからvideoidを取得
    videoid_list = mycode.get_videoid(query=query)

    # videoid_listをもとにvideo_responseを取得する
    video_response_list = mycode.get_video_response_list(videoid_list=videoid_list)

    # videoの情報すべてを入れるdict
    all_video_info = {}

    # ラベルの推定
    pretrained_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    with open("./data/label2id_dic.pickle", "rb") as f:
        label2id_dic = pickle.load(f)
    checkpoint_path = "./data/checkpoint_bert_fine_tuning_youtube.pt"
    model = BertForTokenClassification.from_pretrained(pretrained_path, num_labels=len(label2id_dic))
    model = BERT.model_load_checkpoint(model, checkpoint_path)
    # テキストのトークン化に使用するTokenizerの読み込み
    tokenizer = BERT.tokenizer_for_wardlabeling.from_pretrained(pretrained_path)
    # 形態素レベルまでしか分割しないTokenizer（サブワードレベルまで分割しない）の読み込み
    mecab_tokenizer = transformers.MecabTokenizer(pretrained_path)


    for i in range(len(video_response_list)):
        # video情報の辞書
        video_info = {}
        
        # 動画タイトルの取得
        video_title = video_response_list[i]["items"][0]["snippet"]["title"]
        video_info["video_title"] = video_title
        
        # 動画サムネイルの取得
        # 480×360のサイズにする
        video_caption = video_response_list[i]["items"][0]["snippet"]['thumbnails']["high"]["url"]
        video_info["video_caption"] = video_caption
        
        # 概要欄の取得
        description = video_response_list[i]["items"][0]["snippet"]['description']
        description = description.replace("\n", "<br>")
        description = description.replace("\u3000", "")
        description = description.replace("\200b", "")
        description = description[:450] + "..."
        video_info["description"] = description
        
        # videoidからvideoのURLを作成する
        video_link = "https://youtu.be/" + video_response_list[i]["items"][0]["id"]
        video_info["video_link"] = video_link
        
        # チャンネルURLを取得
        # チャンネルのURL
        channel_link = "https://www.youtube.com/channel/" + video_response_list[i]["items"][0]["snippet"]["channelId"]
        video_info["channel_link"] = channel_link

        # チャンネルのタイトル
        channel_name = video_response_list[i]["items"][0]["snippet"]["channelTitle"]
        video_info["channel_name"] = channel_name
        
        # コメントの取得
        com_list = mycode.print_video_comment(video_response_list[i]["items"][0]["id"], None)


        # チェックポイントからモデルのパラメータの読み込み
        # モデルの定義
        # 事前学習モデルのpath
        # pretrained_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        # with open("./data/label2id_dic.pickle", "rb") as f:
        #     label2id_dic = pickle.load(f)
        # checkpoint_path = "./data/checkpoint_bert_fine_tuning_youtube.pt"
        # model = BertForTokenClassification.from_pretrained(pretrained_path, num_labels=len(label2id_dic))
        # model = BERT.model_load_checkpoint(model, checkpoint_path)

        # コメントの選択
        # 取得したコメントのリストを回してBERTに入れる
        # BERTのラベルで動画に関するコメントと述べられたものだけリストに追加する
        # all_video_infoの中にいれる
        bert_com_list = []
        if com_list is not None:
            for com in com_list:
                if BERT.get_bert_com(comment=com, 
                model=model, 
                tokenizer=tokenizer, 
                mecab_tokenizer=mecab_tokenizer, 
                label2id_dic=label2id_dic) != None:
                    bert_com_list.append(com)
            
        video_info["comment"] = bert_com_list

        
        all_video_info[video_response_list[i]["items"][0]["id"]] = video_info



    # 返値を表示
    return render_template("result.html", 
    query=query, 
    all_video_info=all_video_info)







if __name__ == '__main__':
    app.run(debug=True)
