import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
im = pilimg.open("./images/cat1.webp")

import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.datasets import mnist 
import PIL.Image as pilimg

print(type(mnist) ) 

(train_images, train_labels), (test_images, test_labels)=mnist.load_data() 
print( train_images.shape ) # 28 by 28 이미지가 60000개 있음 

#앞에 천개만 image1.jpg image2.jpg 형태로 저장하기 
for i in range(1000):
    pilimg.fromarray(train_images[i]).save("./images/number/image{}.png".format(i))    

