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
5. check code/train/config.py: dataset/models and other parameters
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
nvidia_arr(labels, probs): 
  将传入的labels和probs concat.先labels再concat，有一方为none则全设为0 （16）
cut_words_from_text: 
  调用 def del_url_at: 分割文本为list，例如：['a', 'test', 'string', '.', 'a', 'test', 'string', '!']
extract_dual_emotion: 
  extract_publisher_emotion，extract_social_emotion，计算得到的gap （270）
extract_publisher_emotion:
  nvidia_arr，nrc_arr，sentiment_score，auxilary_features （54）
extract_social_emotion: 
  每个comment都得到publisher emotion arr，计算mean和max，social_emotion = mean，max （108）
nrc_arr: 
  lexicon and intensity (18)
sentiment_score: 
  调用nltk内置函数返回scores['pos'], scores['neg'], scores['neu'], scores['compound'] (4)
auxilary_features：
  emoticon_arr，symbols_count，sentiment_words_count，pronoun_count，upper_letter_count （16）
emoticon_arr：
  smiling率，frowning率，个数（3）
symbols_count：
  excl, ques, comma, dot, ellip（5）#比率
sentiment_words_count：
  negative and degree（4）
pronoun_count：比率（3）
upper_letter_count：比率（1）
```
##### code/model
MLP / BiGRU / 
##### code/train
master.py: 调用config的参数和train.main()
train.py
eample running:
```
================ [2022-10-25 21:05:12] ================
[Dataset]	RumourEval-19
[Model]	MLP

The hyparameters: 
[Epoch]	50
[Batch Size]	32
[L2 param]	0.01
[Learning Rate]	0.001


Train data: (327, 270), Train label: (327, 3)
Val data: (38, 270), Val label: (38, 3)
Test data: (81, 270), Test label: (81, 3)

2022-10-25 21:05:12.595291: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2022-10-25 21:05:12.603319: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 270)]             0         
_________________________________________________________________
dense (Dense)                (None, 64)                17344     
_________________________________________________________________
dense_1 (Dense)              (None, 48)                3120      
_________________________________________________________________
dense_2 (Dense)              (None, 32)                1568      
_________________________________________________________________
dense_3 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 51        
=================================================================
Total params: 22,611
Trainable params: 22,611
Non-trainable params: 0
_________________________________________________________________
None


-------------------- Train --------------------

Sample Weights when traning: 
None

2022-10-25 21:05:12.792420: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/50
11/11 [==============================] - 1s 62ms/step - loss: 3.3081 - accuracy: 0.2837 - val_loss: 3.0612 - val_accuracy: 0.4211
Epoch 2/50
11/11 [==============================] - 0s 3ms/step - loss: 2.9962 - accuracy: 0.3984 - val_loss: 2.8112 - val_accuracy: 0.4211
Epoch 3/50
11/11 [==============================] - 0s 4ms/step - loss: 2.7447 - accuracy: 0.4902 - val_loss: 2.6301 - val_accuracy: 0.3947
Epoch 4/50
11/11 [==============================] - 0s 3ms/step - loss: 2.5221 - accuracy: 0.4919 - val_loss: 2.4530 - val_accuracy: 0.3947
Epoch 5/50
11/11 [==============================] - 0s 3ms/step - loss: 2.3477 - accuracy: 0.5180 - val_loss: 2.3577 - val_accuracy: 0.3947
Epoch 6/50
11/11 [==============================] - 0s 3ms/step - loss: 2.1815 - accuracy: 0.5322 - val_loss: 2.1816 - val_accuracy: 0.3421
Epoch 7/50
11/11 [==============================] - 0s 4ms/step - loss: 2.0551 - accuracy: 0.5808 - val_loss: 2.0828 - val_accuracy: 0.4211
Epoch 8/50
11/11 [==============================] - 0s 4ms/step - loss: 1.9367 - accuracy: 0.5921 - val_loss: 2.0477 - val_accuracy: 0.3421
Epoch 9/50
11/11 [==============================] - 0s 4ms/step - loss: 1.8441 - accuracy: 0.5686 - val_loss: 1.8877 - val_accuracy: 0.4211
Epoch 10/50
11/11 [==============================] - 0s 4ms/step - loss: 1.7554 - accuracy: 0.6040 - val_loss: 1.8976 - val_accuracy: 0.3421
Epoch 11/50
11/11 [==============================] - 0s 3ms/step - loss: 1.7035 - accuracy: 0.5413 - val_loss: 1.9294 - val_accuracy: 0.3684
Epoch 12/50
11/11 [==============================] - 0s 4ms/step - loss: 1.6084 - accuracy: 0.6030 - val_loss: 1.7919 - val_accuracy: 0.3684
Epoch 13/50
11/11 [==============================] - 0s 4ms/step - loss: 1.5615 - accuracy: 0.6072 - val_loss: 1.8789 - val_accuracy: 0.4474
Epoch 14/50
11/11 [==============================] - 0s 3ms/step - loss: 1.4813 - accuracy: 0.6636 - val_loss: 1.7668 - val_accuracy: 0.3947
Epoch 15/50
11/11 [==============================] - 0s 4ms/step - loss: 1.4554 - accuracy: 0.6266 - val_loss: 1.8216 - val_accuracy: 0.3684
Epoch 16/50
11/11 [==============================] - 0s 3ms/step - loss: 1.4782 - accuracy: 0.5822 - val_loss: 1.6445 - val_accuracy: 0.4474
Epoch 17/50
11/11 [==============================] - 0s 4ms/step - loss: 1.4083 - accuracy: 0.6467 - val_loss: 1.7576 - val_accuracy: 0.3684
Epoch 18/50
11/11 [==============================] - 0s 4ms/step - loss: 1.3633 - accuracy: 0.6627 - val_loss: 1.8192 - val_accuracy: 0.2895
Epoch 19/50
11/11 [==============================] - 0s 3ms/step - loss: 1.3279 - accuracy: 0.6820 - val_loss: 1.9079 - val_accuracy: 0.3421
Epoch 20/50
11/11 [==============================] - 0s 4ms/step - loss: 1.3251 - accuracy: 0.6857 - val_loss: 1.8722 - val_accuracy: 0.4211
Epoch 21/50
11/11 [==============================] - 0s 4ms/step - loss: 1.2986 - accuracy: 0.6845 - val_loss: 1.6953 - val_accuracy: 0.3684
Epoch 22/50
11/11 [==============================] - 0s 3ms/step - loss: 1.2360 - accuracy: 0.7269 - val_loss: 1.6569 - val_accuracy: 0.4474
Epoch 23/50
11/11 [==============================] - 0s 5ms/step - loss: 1.2275 - accuracy: 0.7301 - val_loss: 1.9224 - val_accuracy: 0.2895
Epoch 24/50
11/11 [==============================] - 0s 5ms/step - loss: 1.2206 - accuracy: 0.6781 - val_loss: 1.7350 - val_accuracy: 0.3421
Epoch 25/50
11/11 [==============================] - 0s 3ms/step - loss: 1.1743 - accuracy: 0.7354 - val_loss: 1.6700 - val_accuracy: 0.4474
Epoch 26/50
11/11 [==============================] - 0s 3ms/step - loss: 1.1350 - accuracy: 0.7611 - val_loss: 1.6888 - val_accuracy: 0.3947

-------------------- val --------------------


TEST_sz: 38
fake_sz: 19
real_sz: 10
unverified_sz: 9

Accuracy: 0.447

              precision    recall  f1-score   support

        fake      0.600     0.316     0.414        19
        real      0.421     0.800     0.552        10
  unverified      0.333     0.333     0.333         9

   micro avg      0.447     0.447     0.447        38
   macro avg      0.451     0.483     0.433        38
weighted avg      0.490     0.447     0.431        38
 samples avg      0.447     0.447     0.447        38




-------------------- test --------------------


TEST_sz: 81
fake_sz: 40
real_sz: 31
unverified_sz: 10

Accuracy: 0.457

              precision    recall  f1-score   support

        fake      0.500     0.375     0.429        40
        real      0.429     0.677     0.525        31
  unverified      0.500     0.100     0.167        10

   micro avg      0.457     0.457     0.457        81
   macro avg      0.476     0.384     0.373        81
weighted avg      0.473     0.457     0.433        81
 samples avg      0.457     0.457     0.457        81
```
