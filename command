python3 train.py train \
  --gpu 0 \
  --optim.num_epoch 100 \
  --optim.batch-size 16 \
  --data.base_action_dim 5 \
  --data.data-key proprio tactile \
  --train_data /home/lifan/Documents/GitHub/minbc/data/train \
  --test_data /home/lifan/Documents/GitHub/minbc/data/test \
  --output_name my_tactile_robot_test