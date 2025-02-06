#import torch
#print(torch.cuda.is_available())  # Should return True
#print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
import tensorflow as tf

print(tf.__version__)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # Use the first GPU
    print("GPU is set for use.")
else:
    print("No GPU detected.")