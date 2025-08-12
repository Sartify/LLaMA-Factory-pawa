python src/export_model.py \
	--model_name_or_path outputs/pawa-zulu-pt/checkpoint-3000 \
	--template default \
	--finetuning_type lora \
	--checkpoint_dir ckpoint \
	--export_dir 7b-ft
