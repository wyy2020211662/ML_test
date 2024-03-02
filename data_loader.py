
#本地依赖包 gcommand_loader
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
from gcommand_loader import GCommandLoader
import os
import tensorflow as tf
image_size = [320,320]
import cv2
def load_image(image_path, image_size):
  """Loads an image, and returns numpy.ndarray.

  Args:
    image_path: str, path to image.
    image_size: list of int, representing [width, height].

  Returns:
    image_batch: numpy.ndarray of shape [1, H, W, C].
  """
  input_data = tf.io.gfile.GFile(image_path, 'rb').read()
  image = tf.io.decode_image(input_data, channels=3, dtype=tf.uint8)
  image = tf.image.resize(
      image, image_size, method='bilinear', antialias=True)
  return tf.expand_dims(tf.cast(image, tf.uint8), 0).numpy()

def load_data_set(data_set,global_count):
    data=[]
    count = 0
    print(f"{data_set}数据加载中...")
    if data_set == "gsc":
        test_dataset = GCommandLoader("data/gsc/test", window_size=.02, window_stride=.01,
                              window_type='hamming', normalize=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    elif data_set =="object_detection":
        test_dataset=[]
        test_loader=[]
        results=os.listdir('/home/pi/automl/efficientdet/coco_val2017')
        for image_path in results[:100]:
            test_loader.append([load_image("/home/pi/automl/efficientdet/coco_val2017/"+image_path, image_size),1])
        
    elif data_set =="fall_down_detection":
        test_dataset=[]
        test_loader=[]
        results=os.listdir('/home/pi/automl_for_fall_dectection/efficientdet/testdata/images')
        for image_path in results[:100]:

            test_loader.append([load_image('/home/pi/automl_for_fall_dectection/efficientdet/testdata/images/'+image_path, image_size),1])
    elif data_set=="food_recognition":
        data_dir = '/home/pi/FoodRecognition/'
        # 定义数据转换，包括图像大小调整from memory_profiler import profile、归一化等
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.Food101(root=data_dir,split='test',transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    else:
        test_dataset=[]
        test_loader=[]
        testfolderPath = "/home/pi/fingercounter/TestFingerImages"
        test_loader.append([cv2.imread('/home/pi/fingercounter/TestFingerImages/5.jpg'),1])
        

    print(f"{data_set}数据加载完毕")
    for inputs, labels in test_loader:
        count+=1
        data.append([inputs, labels])
        if count >= global_count:
            break
    del test_loader,test_dataset
    return data
