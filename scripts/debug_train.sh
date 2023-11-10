# train model
export COCO_2017_DIR=/opt/datasets/COCO_2017/raw/Images/train2017
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client /root/anaconda3/envs/lisa/bin/deepspeed --master_port=24999 one_model/train/trainer.py --conf  configs/train/config_7B.yaml --exp_name debug-one-model-7b --dataset_dir /opt/product/dataset --ds_config_path configs/train/zero2.json
