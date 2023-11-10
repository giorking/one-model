# 简介
openseed 使用transformer 统一各种分割任务，包括语义分割，实例分割，全景分割，并且在开放数据集上取得了不错的效果。

## 安装使用
### 下载openseed 代码

```
git clone git@git.xiaojukeji.com:qiudanwang_i/OpenSeeD.git
```
### 安装依赖
#### 编译deformed  ops
```
cd openseed/body/encoder/ops
python setup.py build install --user
```
#### 代码使用
由于openseed 不是一个标准的python package，所以需要将openseed的根目录加入到python path中
```
export PYTHONPATH=/path/to/openseed:$PYTHONPATH
```
## 测试全景分割
```
cd openseed
sh test.sh
```

## 模型weight
```
cos://vision-1302847974/ckpts/one_model/openseed.tar.gz
```

