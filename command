python3 train.py train \
  --gpu 0 \
  --optim.num_epoch 1000 \
  --optim.batch-size 16 \
  --data.base_action_dim 5 \
  --data.data-key proprio tactile \
  --train_data /home/lifan/Documents/GitHub/minbc/data/train \
  --test_data /home/lifan/Documents/GitHub/minbc/data/test \
  --output_name /home/lifan/Documents/GitHub/minbc/outputs/test_00002