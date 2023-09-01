# instructions for training RNNs

## using pytorch lightning, etc.

where the code lies:
- the model code is in `mocapact/distillation/model.py`, and is fully contained in the `RNNPolicy` class.
    - the code is extensively documented
    - the `RNNPolicy.training_step` does a forward step on one batch and returns the loss. this is then used by the trainer (see below) to optimize the weights. this only works with pytorch.
- the dataset is defined in `mocapact/distillation/dataset.py` within the `ExpertDatasetRNN` class.
    - the dataset simply loads data from the hdf5 files (these are "expert datasets" from mocapact, so not the original mocap data)
    - you can index the dataset to get different episodes (rollouts). by default, the first 200 episodes are initialized at the start, and the next 200 episodes are initialized at random points in the snippet.
- the setup and training loop is defined in `mocapact/distillation/trainer.py`
    - that file can be run directly from the command line, it accepts all the arguments that you may wish to change in the training setup. 
    - an example setup for how to run the file with all its arguments is shown in `script_train_rnn.sh`, run it with `source script_train_rnn.sh` from the command line (make sure to have the right python environment active).
    - the trainer proceeds as follows
        1. setup dataset, create dataloader (for batching, using a custom collate function to pad episodes of different sequence lengths)
        2. (it also sets up validation things but we don't need them, so you can ignore that)
        3. setup policy model
        4. create callbacks for: checkpointing (saves the top-k models), training logs (csv file and tensorboard), validation, etc.
        5. in the very end, the `PytorchLightning.Trainer` object is created which handles the training loop. in the call `trainer.fit`, the training loop is executed with the `policy` network and the `train_loader` data.
    - once the training runs, you can run tensorboard to see how the training is going (loss, mse, time), using `source script_run_tensorboard.sh`

## also using JAX

see notebook [nb_train_rnn_jax.ipynb](./nb_train_rnn_jax.ipynb)
