import mycode

query = "なにわ男子"
# クエリからvideoidを取得
videoid_list = mycode.get_videoid(query=query)

# videoid_listをもとにvideo_responseを取得する
video_response_list = mycode.get_video_response_list(videoid_list=videoid_list)

# videoの情報すべてを入れるdict
all_video_info = {}


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
    video_info["comment"] = com_list
    
    all_video_info[video_response_list[i]["items"][0]["id"]] = video_info

    print(video_info)