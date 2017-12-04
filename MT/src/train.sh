#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH -t 0

module load cuda-8.0 cudnn-8.0-5.1

export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/cudnn-8.0/include:$CPATH
export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH

python2 train_bucket.py -task Multi-MT -data iwslt15_de_fr_en -adv_lambda 8. -rb_vec_size 3 -epochs 20 -disc_size 256 -adv_update_freq 1 -rnn_size 512 -num_rb_bin 2 -disc_type RNN -separate_encoder 1 -use_rb_emb 0 -gpus 1 -disc_bi_dir 0
