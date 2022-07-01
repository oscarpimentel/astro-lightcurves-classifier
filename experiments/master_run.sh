#!/bin/bash
clear
SECONDS=0
run_script(){
	echo "$1"; eval "$1";
}
# *************************************************************************************************************************
# mc_gpu="--mc p_attn_model --gpu 0 --precompute_only 1"

### serial
# mc_gpu="--mc s_rnn_models --gpu 0" # RNN
mc_gpu="--mc s_attn_model --gpu 2" # ATTN

# extras
# mc_gpu="--mc s_attn_models_te --gpu 0"
# mc_gpu="--mc s_attn_models_heads --gpu 0"
# mc_gpu="--mc s_attn_models_dummy --gpu 1"
# mc_gpu="--mc s_rnn_extra_models --gpu 0"
# mc_gpu="--mc s_attn_extra_models --gpu 0"

### parallel
# mc_gpu="--mc p_rnn_models --gpu 1" # RNN
# mc_gpu="--mc p_attn_model --gpu 0" # ATTN

# extras
# mc_gpu="--mc p_attn_models_te --gpu 1"
# mc_gpu="--mc p_attn_models_heads --gpu 1"
# mc_gpu="--mc p_attn_models_dummy --gpu 1"
# mc_gpu="--mc p_rnn_extra_models --gpu 1"
# mc_gpu="--mc p_attn_extra_models --gpu 2"

extras=""\
"--only_perform_exps 0 "\
"--perform_slow_exps 0 "\
"--bypass_pre_training 0 "\
"--batch_size 202 "\
"--preserved_band . "\
"--bypass_synth 0 "\
"--bypass_prob 0 "\
"--ds_prob 0.1 "\

for mid in {1000..1005}; do # 1000..1005
	for kf in {0..4}; do # 0..4
		run_script "python train_deep_models.py --mid $mid --kf $kf $mc_gpu $extras"
		:
	done
done

# *************************************************************************************************************************
mins=$((SECONDS/60))
echo echo "time elapsed=${mins} [mins]"
