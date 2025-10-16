CUDA_VISIBLE_DEVICES=1 \
	accelerate launch \
	--config_file pawa/configs/pretrain-eng-swa/acc.yaml \
	src/train.py pawa/configs/pretrain-eng-swa/pawa_llamafactory.yaml
