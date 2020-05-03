import numpy as np
from AlexNet import alexnet
WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 3
MODEL_NAME = 'T-rex{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

def main()
  model = alexnet(WIDTH, HEIGHT, LR)
  #Setup the training data:

  train_data = np.load('training_data.npy', allow_pickle=True)

  train = train_data[:-500]
  test = train_data[-500:]

  X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
  Y = [i[1] for i in train]

  test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
  test_y = [i[1] for i in test]
  #Now we can actually train the model with:

  model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

  # tensorboard --logdir=foo:C:/Users/H/Desktop/ai-gaming/log

  model.save(MODEL_NAME)
main()
