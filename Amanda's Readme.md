### Dataset
1. Dir: WWW2021/dataset/dataset_name/
2. 3 json files in each dataset folder with name: train, val, test
3. format:
```
[{"commets":[string, string, string,...],
  "content": string,
  "content_emotions_labels":{"anger": 0.0,
            "anticipation": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 1.0,
            "sadness": 0.0,
            "surprise": 0.0,
            "trust": 0.0}
  "content_emotions_probs":{"anger": 0.07779739797115326,
            "anticipation": 0.04301946610212326,
            "disgust": 0.055976446717977524,
            "fear": 0.012341899797320366,
            "joy": 0.29545190930366516,
            "sadness": 0.09565771371126175,
            "surprise": 0.0010137018980458379,
            "trust": 0.1391817033290863},
  "id":string,
  "label":string, # fake, real or unverified.
  "comments100_emotions_labels_mean_pooling": format same as content emotions labels, #最早发出的100条评论
  "comments100_emotions_labels_max_pooling": format same as content emotions labels,
  "comments100_emotions_probs_mean_pooling": format same as content emotions probs,
  "comments100_emotions_probs_max_pooling": format same as content emotions probs
  },...]
```

### Workflow
1. cd code/preprocess
2. python output_of_labels
3. python input_of_emotions
4. python input_of_semantics
5. check code/train/config.py
6. cd code/train
7. python master.py

### Codes
##### code/preprocess
1. output_of_labels: 提取datasets中的labels为（n,3）array，存到code/preprocess/data/dataset_name/labels的npy文件中
2. input_of_emotions调用emotion/extract_emotion_en: 将content和comment的情感信息提取为（n,270）array, 存到code/preprocess/data/dataset_name/emotions的npy文件中
3. input_of_semantics: 用standford Glove to get embeddings.设定embedding matrix shape with CONTENT_WORDS & EMBEDDING_DIM
存到 code/preprocess/data/dataset_name/semantics的npy文件中
##### code/emotion
extract_emotion_en: preprocess时调用
```
def nvidia_arr(labels, probs): 将传入的labels和probs concat.先labels再concat，有一方为none则全设为0
def cut_words_from_text 调用 def del_url_at: 分割文本为list，例如：['a', 'test', 'string', '.', 'a', 'test', 'string', '!']
def extract_dual_emotion = extract_publisher_emotion，extract_social_emotion，计算得到的gap
def extract_publisher_emotion = nvidia_arr，nrc_arr，sentiment_score，auxilary_features
def extract_social_emotion 每个comment都得到publisher emotion arr，计算mean和max，social_emotion = mean，max
```
##### code/model

##### code/train
