from apiclient.discovery import build
from apiclient.errors import HttpError
import requests
import re

# API情報
DEVELOPER_KEY = 'PASS'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
URL = 'https://www.googleapis.com/youtube/v3/'


# 関数を作成する
# クエリを入力したらそのクエリでおすすめの動画10件のvideoidを取得できる
def get_videoid(query):
    # 検索結果からvideoidだけ取得する

    youtube = build(
        YOUTUBE_API_SERVICE_NAME, 
        YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY
        )
    
    # json型で保存されている
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=5
        ).execute()
    
    # search_responceからvideoidを取得する
    video_id_list = []
    for i in range(len(search_response["items"])):
        # videoIdがあったらvideoidをリストに追加する
        if "videoId" in search_response["items"][i]["id"].keys():
            video_id_list.append(search_response["items"][i]["id"]['videoId'])

    return video_id_list

# videoidが与えられたときにその動画の情報を返す
def get_video_info(videoid):

    youtube = build(
        YOUTUBE_API_SERVICE_NAME, 
        YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY
        )
    
    # ビデオの情報を取得
    video_response = youtube.videos().list(
        part='snippet',
        id=videoid,
        ).execute()
    return video_response

# video_responseをリスト上で返すもの
def get_video_response_list(videoid_list):

    youtube = build(
        YOUTUBE_API_SERVICE_NAME, 
        YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY
        )

    video_response_list = []
    for video_id in videoid_list:
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id,
            ).execute()
        video_response_list.append(video_response)
    return video_response_list

# videoidからコメントを取得する
def print_video_comment(video_id, next_page_token):
    params = {
        'key': DEVELOPER_KEY,
        'part': 'snippet',
        'videoId': video_id,
        'order': 'relevance',
        'textFormat': 'plaintext',
        'maxResults': 20,
      }
    if next_page_token is not None:
        params['pageToken'] = next_page_token
    response = requests.get(URL + 'commentThreads', params=params)
    resource = response.json()

    text_list = []
    if "error" not in resource.keys():
        for comment_info in resource['items']:
            # コメント
            text = comment_info['snippet']['topLevelComment']['snippet']['textDisplay']
            text = text.replace("\n", "<br>")
            text = text.replace("\u3000", " ")
            
            if re.search(r'[ぁ-んァ-ン]', text):
                text_list.append(text)
                

        return text_list
        


