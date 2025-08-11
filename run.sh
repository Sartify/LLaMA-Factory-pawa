CUDA_VISIBLE_DEVICES=0,1 \
	accelerate launch \
	--config_file pawa/configs/pretrain-zulu/acc.yaml \
	src/train.py pawa/configs/pretrain-zulu/pawa_llamafactory.yaml
