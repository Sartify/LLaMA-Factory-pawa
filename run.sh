CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	accelerate launch \
	--config_file pawa/configs/acc.yaml \
	src/train.py pawa/configs/pawa_llamafactory.yaml
