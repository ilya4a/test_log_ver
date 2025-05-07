# config.py
import torch

BATCH_SIZE = 256
LEARNING_RATE = 0.1
EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
