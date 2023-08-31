python -m mocapact.distillation.evaluate \
  --policy_path multiclip/walkrun_rnn/model/last.ckpt `# e.g., multi_clip/all/eval/train_rsi/best_model.ckpt` \
  --clip_snippets CMU_016_15-0-127,CMU_016_55-0-47 \
  --n_workers 1 `# e.g., 8` \
  --device cpu `# e.g., cuda` \
  --n_eval_episodes 0 `# 1000, set to 0 to just run the visualizer` \
  --act_noise 0. \
  --noalways_init_at_clip_start \
  --eval_save_path multi_clip_evaluation.npz \
  --visualize \
  --ghost_offset 1.
