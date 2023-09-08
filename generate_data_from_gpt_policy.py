import numpy as np
import subprocess
import json#
import os
from tqdm import trange
# define parameters for calling the gpt policy 

REMOVE_TEMP_FILES = True

num_features = 206
num_actions = 56

# number of sequences to generate
num_seqs = 100
# minimum length of each sequence
min_length = 140


dataset = "CMU_016_55"

# Define the command as a single string with parameters
command = [
        f"python -m mocapact.distillation.motion_completion_data_gen \
        --policy_path ./gpt.ckpt \
        --nodeterministic \
        --ghost_offset 1 \
        --expert_root ../data/experts \
        --max_steps 500 \
        --always_init_at_clip_start \
        --prompt_length 35 \
        --termination_error_threshold 0.3 \
        --min_steps 80 \
        --device cpu \
        --clip_snippet {dataset}\
        --num_seqs {num_seqs} \
        --min_length {min_length}"
    ]

# Run the gpt policy to generate data
subprocess.call(command,
                    shell=True,
                    stderr=subprocess.STDOUT)

for run in range(0,num_seqs):
    
    # load json data from observation and action files
    
    # load observation data
    with open(f'.observations_{run}.json') as f:
        obs_data = json.load(f)
        
        
    observations = []
    for i in range(len(obs_data['observations'])):
        obs_data_t = obs_data['observations'][i]
        for k, v in obs_data_t.items():
            observations.append(v[0]) 
        
    observations = np.array(observations)[:-1]

    np.savez(f".observations_{run}.npz", observations=observations)
    os.remove(f".observations_{run}.json")

    with open(f'.actions_{run}.json') as f:
        act_data = json.load(f)

    actions = []
    for i in range(len(act_data['actions'])):
        act_data_t = act_data['actions'][i]
        for k, v in act_data_t.items():
            print(k)
            actions.append(act_data_t[k]) 
        
    actions = np.array(actions)[:-1]
    np.savez(f".actions_{run}.npz", actions=actions)
    os.remove(f".actions_{run}.json")


features = []
actions = []

for i in range(num_seqs):
    obs = np.load(f".observations_{i}.npz",allow_pickle=True)
    features.append(np.array(obs["observations"]).reshape(-1,num_features))
    
    acts = np.load(f".actions_{i}.npz",allow_pickle=True)
    actions.append(acts["actions"])

max_len = 0
for i in range(num_seqs):
    len = np.amax(features[i].shape[0])
    max_len = max(len,max_len)
    
seq_len = []
for i in range(num_seqs):
    pad = np.zeros((max_len,num_features))
    flen = features[i].shape[0]
    pad[:flen,:] = features[i]
    features[i] = pad
    seq_len.append(flen)
    
    pad = np.zeros((max_len, num_actions))
    flen = actions[i].shape[0]
    pad[:flen,:] = actions[i]
    actions[i] = pad
    
features = np.array(features).reshape(num_seqs,-1,num_features)
actions = np.array(actions).reshape(num_seqs,-1, num_actions)
seq_len = np.array(seq_len).reshape(num_seqs)
np.savez("gpt_policy_data_running_long.npz", features=features, actions=actions, seq_len=seq_len)

if REMOVE_TEMP_FILES:
    for i in range(num_seqs):
        os.remove(f".observations_{i}.npz")
        os.remove(f".actions_{i}.npz")