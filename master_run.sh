#!/bin/bash
reset
SECONDS=0
run_python_script(){
	now=$(date +"%T")
	echo -e "\e[7mrunning script ($now)\e[27m python $1"
	eval "python $1"  # to perform serial runs
	# eval "python $1 > /dev/null 2>&1" &  # to perform parallel runs
}
intexit(){
    kill -HUP -$$
}
hupexit(){
    echo
    echo "Interrupted"
    exit
}
trap hupexit HUP
trap intexit INT
echo -e "\e[7mrunning master_run... (ctrl+c to interrupt)\e[27m $1"
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# mc_gpu="--mc p_attn_model --gpu 0 --precompute_only 1"

### serial
# mc_gpu="--mc s_rnn_models --gpu 0" # RNN
# mc_gpu="--mc s_attn_model --gpu 2" # ATTN

# extras
# mc_gpu="--mc s_attn_models_te --gpu 0"
# mc_gpu="--mc s_attn_models_heads --gpu 0"
# mc_gpu="--mc s_attn_models_dummy --gpu 1"
# mc_gpu="--mc s_rnn_extra_models --gpu 0"
# mc_gpu="--mc s_attn_extra_models --gpu 0"

### parallel
# mc_gpu="--mc p_rnn_models --gpu 1" # RNN
mc_gpu="--mc p_attn_model --gpu 1" # ATTN

# extras
# mc_gpu="--mc p_attn_models_te --gpu 1"
# mc_gpu="--mc p_attn_models_heads --gpu 1"
# mc_gpu="--mc p_attn_models_dummy --gpu 1"
# mc_gpu="--mc p_rnn_extra_models --gpu 1"
# mc_gpu="--mc p_attn_extra_models --gpu 2"

extras=""\
"--only_perform_exps 1 "\
"--perform_slow_exps 1 "\
"--bypass_pre_training 0 "\
"--batch_size 203 "\
"--preserved_band . "\
"--bypass_synth 0 "\
"--bypass_prob 0 "\
"--ds_prob 0.1 "\

for mid in {1000..1005}; do # 1000..1005
	for kf in {0..4}; do # 0..4
		run_python_script "train_deep_models.py --mid $mid --kf $kf $mc_gpu $extras"
		:
	done
done
wait

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
mins=$((SECONDS/60))
echo -e "\e[7mtime elapsed=${mins}[mins]\e[27m"
