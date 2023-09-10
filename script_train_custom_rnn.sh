# n_workers, temperature, model.config.embed_size
# --eval_mode train_rsi \
# --eval_mode val_start \
# --eval_mode val_rsi \
# --eval_mode clips_start \
# --eval_mode clips_rsi \
# --nomodel.config.squash_output \
# --model.config.activation_fn torch.nn.Tanh \
# --model.config.embedding_kl_weight 0.1 \
# --model.config.embedding_correlation 0. \
# --model.config.embed_size 60 \
# --clip_len_upsampling \
# eval
# --eval_mode train_start \
# --validation_freq 10000 \
# --val_start_rollouts 20 \
# --val_rsi_rollouts 20 \
# --eval.freq 10000 \
# --eval.n_episodes 1000 \
# --eval.n_workers 16
# --extra_clips CMU_016_15,CMU_016_55 \
# batch size used to be 256 for non-RNN training
# n_workers used to be 8
#   --load_path multiclip/walkrun_rnn/model/last.ckpt \
python -m mocapact.distillation.train \
	--model mocapact/distillation/config.py:rnn_reference \
	--train_dataset_paths ../data/rollouts/CMU_016_15.hdf5,../data/rollouts/CMU_016_36.hdf5 \
	--dataset_metrics_path ../data/rollouts/dataset_metrics.npz \
	--output_root custom_rnn/walkjog \
	--batch_size 32 \
	--n_workers 8 \
	--learning_rate 0.0005 \
	--n_steps 100000 \
	--max_grad_norm 1. \
	--normalize_obs \
	--save_top_k 10 \
	--noinclude_timestamp \
	--temperature 4. \
	--noadvantage_weights
