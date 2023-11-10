# one-model
one-model for all mllm

## 部署环境
要求python 3.8以上
### 下载sam代码并安装
```bash
git clone --branch support-text-embdings git@git.xiaojukeji.com:qiudanwang_i/segment-anything.git
cd segment-anything && pip install -e . 
```

### 下载one-model代码并安装依赖
```bash
git clone --branch main git@git.xiaojukeji.com:qiudanwang_i/one-model.git
cd one-model && pip install -e .
```


## 数据准备
数据主要依赖分割数据和vqa数据，从这里下载 [链接](https://cosbrowser.cloud.tencent.com/share/?id=MdWbh0iaNaW0Nfkfb5n7E61e&token=mgppOXavj1Cf6QYGvXwBYhhW0vucArl1CCF%2BChS%2F31TbYjOFOd5Chrw%2BqWrjbnfS)  提取码: f9aa32，解压之后目录结构组织成如下格式：

```
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   └── vlpart
│       ├── paco
│       │   └── annotations
│       └── pascal_part
│           ├── train.json
│           └── VOCdevkit
```

## 训练
参考 train.sh 脚本，训练的时候需要指定模型的版本，13B或者7B的， 数据集的目录， 和结果文件的目录
```bash
sh scripts/train.sh
```

## 预测
参考 infer.sh, 可以设置int8 或者 int4量化
```bash
sh scripts/infer.sh 
```

## 预训练模型下载
从云盘下载预训练模型13B 放到hub目录下， ~/.cache/huggingface/hub
```
llama2-13b

https://vision-1302847974.cos.ap-shanghai.myqcloud.com/ckpts/models--xinlai--LISA-13B-llama2-v1-explanatory.tar.gz?q-sign-algorithm=sha1&q-ak=AKIDCqGzUiq9XjKfolFO5arv490Vw7KIpWxb&q-sign-time=1695692866;1695700066&q-key-time=1695692866;1695700066&q-header-list=&q-url-param-list=&q-signature=20acd5785a94f85994da86eec16b668b346ad086

clip-large
https://vision-1302847974.cos.ap-shanghai.myqcloud.com/ckpts/models--openai--clip-vit-large-patch14.tar.gz?q-sign-algorithm=sha1&q-ak=AKIDCqGzUiq9XjKfolFO5arv490Vw7KIpWxb&q-sign-time=1695692897;1695700097&q-key-time=1695692897;1695700097&q-header-list=&q-url-param-list=&q-signature=0802b099a4e82e7941748afa5e3640ea2a4cacda

```
