# train model
export COCO_2017_DIR=/opt/datasets/COCO_2017/raw/Images/train2017
CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=true deepspeed --master_port=24999 one_model/train/trainer.py --conf  configs/train/config_7B.yaml --exp_name one-model-7b --ds_config_path configs/train/zero2.json
