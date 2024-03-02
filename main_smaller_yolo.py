
r"""Run TF Lite model."""
"hello!!!!!!!!!!!!"
#本地依赖包 inference tf2 data_loader HandTrackingModule video_player vlc
import inference

from absl import app
from absl import flags
import os
from PIL import Image
import time  
import tensorflow as tf
from tf2 import label_util
import random
import torch
from data_loader import  load_data_set
import numpy as np
import HandTrackingModule as htm
import video_player as vp 
import vlc
import cv2
import torchvision.transforms as transforms
import six
import threading

# 全局变量 BGR
font_color=(0,255,0)

img_size=320
#视频源
video_frame = np.ones([img_size,img_size,3],dtype=np.uint8)

#监控画面
display_frame = np.ones([img_size,img_size,3],dtype=np.uint8)
#目标检测画面
display_frame_object = np.ones([img_size,img_size,3],dtype=np.uint8)
#摔倒检测画面
display_frame_fall = np.ones([img_size,img_size,3],dtype=np.uint8)
#菜品检测画面
display_frame_food = np.ones([img_size,img_size,3],dtype=np.uint8)

#手势检测画面
display_frame_gesture = np.ones([img_size,img_size,3],dtype=np.uint8)

#语音识别结果
display_frame_gsc = np.ones([img_size,img_size,3],dtype=np.uint8)

g_flag=0
exit_sign=False
lock = threading.Lock()

#将监控画面、目标检测画面、摔倒检测画面、菜品检测画面拼接在一起
def display():
    print("play....")
    global display_frame ,display_frame_fall,display_frame_object,display_frame_gesture,display_frame_food,exit_sign,display_frame_gsc
    while True:
        time.sleep(0.1)
        if display_frame is not None and  display_frame_fall is not None:
            combined_images = np.concatenate((display_frame ,display_frame_fall,display_frame_object,display_frame_gesture,display_frame_food,display_frame_gsc),axis=1)
            cv2.imshow("Video Capture",combined_images )
        if cv2.waitKey(1) & 0xFF == ord('q') or exit_sign == True:
            break
    cv2.destroyAllWindows()
    print('display done')

#摄像头捕获视频
def capture_video(camera_index=0):
    global video_frame,display_frame
    global exit_sign
    wCam, hCam = img_size, img_size

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while exit_sign==False:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break
            
            video_frame = frame.copy()
            text='camera'
            display_frame =  frame.copy()
            cv2.putText(display_frame,text,(0,30),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)
              
    finally:
        cap.release()
        print('capture_video done')
#摔倒检测和目标检测的数据预处理
def load_image_camera(cv_img, image_size):
  """Loads an image, and returns numpy.ndarray.

  Args:
    image_path: str, path to image.
    image_size: list of int, representing [width, height].

  Returns:
    image_batch: numpy.ndarray of shape [1, H, W, C].
  """
  img_cv_ecd = cv2.imencode('.jpeg',cv_img)[1]
  data_encode = np.array(img_cv_ecd)
  string_encode = data_encode.tobytes()
  image = tf.io.decode_image(string_encode , channels=3, dtype=tf.uint8)
  image = tf.image.resize(
      image, image_size, method='bilinear', antialias=True)
  return tf.expand_dims(tf.cast(image, tf.uint8), 0).numpy()


#将摔倒检测和目标检测的结果显示到对应的视频帧上
def create_visualized_image(image, prediction, model_type):
  global display_frame_object
  global display_frame_fall
  """Saves the visualized image with prediction.

  Args:
    image: numpy.ndarray of shape [H, W, C].
    prediction: numpy.ndarray of shape [num_predictions, 7].
    output_path: str, output image path.
  """

  if model_type=='object_detection':
    output_image = inference.visualize_image_prediction(
      image,
      prediction,
      label_map='coco')
    text='object_detection'
    with lock:
      display_frame_object=cv2.cvtColor((output_image),cv2.COLOR_RGB2BGR)
      cv2.putText( display_frame_object,text,(0,30),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)
  elif model_type=='fall_detection':
    output_image = inference.visualize_image_prediction(
      image,
      prediction,
      label_map='fall')
    text='fall_detection'
    with lock:
      display_frame_fall=cv2.cvtColor((output_image),cv2.COLOR_RGB2BGR)
      cv2.putText( display_frame_fall,text,(0,30),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)

#运行tflite模型
class TFLiteRunner:
  """Wrapper to run TFLite model."""
 # @profile
  def __init__(self, model_path):
    """Init.

    Args:
      model_path: str, path to tflite model.
    """
    self.interpreter = tf.lite.Interpreter(model_path=model_path)
    self.interpreter.allocate_tensors()
    self.input_index = self.interpreter.get_input_details()[0]['index']
    self.output_index = self.interpreter.get_output_details()[0]['index']

  def run(self, image):
    """Run inference on a single images.

    Args:
      image: numpy.ndarray of shape [1, H, W, C].

    Returns:
      prediction: numpy.ndarray of shape [1, num_detections, 7].
    """
    self.interpreter.set_tensor(self.input_index, image)
    self.interpreter.invoke()
    return self.interpreter.get_tensor(self.output_index)


#手势检测
def detect(detector,epoches,thread_name,player):
    
    print(f"{thread_name},start...")
    global video_frame,g_flag,display_frame_gesture
    tipIds = [4, 8, 12, 16, 20]
    sum_time=0
    while True:
        if video_frame is not None:
            break
    
    for i in range (epoches):

        img = video_frame.copy()
        start=time.time()
        img = detector.findHands(img)
        text='gesture_recognition'
        display_frame_gesture=img
        cv2.putText( display_frame_gesture,text,(0,30),cv2.FONT_HERSHEY_COMPLEX,0.8,font_color,1)
        lmList = detector.findPosition(img, draw=False)
        sum_time +=time.time()-start


        if len(lmList) != 0:
            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            print(lmList)
            
            if totalFingers== 5 and g_flag ==0:
                player.pause()
                with lock:
                    g_flag = 1
                    cv2.putText( display_frame_gesture,"暂停",(0,50),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)                          
                print("Pause")
            elif totalFingers== 3 and g_flag ==1:
                player.resume() 
                with lock:
                    g_flag = 0   
                    cv2.putText( display_frame_gesture,"播放",(0,50),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)   
                print("Play")             
            print(totalFingers)
    print(f"{thread_name},exit...sum infer time : {sum_time},avg time: {sum_time/epoches}")

#摔倒检测
def fall_down_infer(runner,size,epoches,thread_name):
    
    print(f"{thread_name},start...")
    global video_frame

    min_score_thresh = .5
    count = 0
    sum_time=0
    while video_frame.all()==None:
        print("video_frame is None")
    
    for i in range (epoches):
        img = video_frame.copy()
        
        start=time.time()
        local_img=load_image_camera(img,size)
        prediction = runner.run(local_img)
        sum_time +=time.time()-start
        create_visualized_image(local_img[0], prediction[0], "fall_detection")
    print(f"{thread_name},exit...sum infer time : {sum_time},avg time: {sum_time/epoches}")
#

#目标检测
def object_infer(runner,size,epoches,thread_name,player):
    print(f"{thread_name},start...")
    d_flag=0
    global video_frame,g_flag
    min_score_thresh = .5
    count = 0
    sum_time=0
    
    for i in range (epoches):
        img = video_frame.copy()
        local_img=load_image_camera(img,size)
        start=time.time()
        prediction = runner.run(local_img)
        sum_time +=time.time()-start
        boxes = prediction[0][:, 1:5]
        classes = prediction[0][:, 6].astype(int)
        scores = prediction[0][:, 5]
        label_map = label_util.get_label_map(None or 'coco')
        category_index = {k: {'id': k, 'name': label_map[k]} for k in label_map}
        return_classes=[]
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                if classes[i] in six.viewkeys(category_index):
                  class_name = category_index[classes[i]]['name']
                  return_classes.append(class_name)
        
        if "person" not in return_classes and d_flag ==0 and g_flag ==0:
            player.pause()
            d_flag = 1
            print("暂停")
        elif "person"  in return_classes  and d_flag ==1 and g_flag ==0:
            player.resume()  
            d_flag = 0   
            print("恢复")
        create_visualized_image(local_img[0], prediction[0], "object_detection")
        
        
    print(f"{thread_name},exit...sum infer time : {sum_time},avg time: {sum_time/epoches}")

def iou(box1, box2):
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # 计算交集区域的面积
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # 计算IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# 抑制非极大值方法
def non_max_suppression(boxes, scores, threshold=0.8):
    # 创建一个用于存储保留的边界框的列表
    keep = []
    # 对得分进行排序
    order = scores.argsort()[::-1]
    # 循环直到所有边界框都被检查
    while order.size > 0:
        # 将当前最大得分的边界框添加到keep中
        i = order[0]
        keep.append(i)
        # 计算剩余边界框与当前边界框的IoU
        ious = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        # 找到与当前边界框IoU小于阈值的边界框
        inds = np.where(ious <= threshold)[0]
        # 更新order，只保留那些与当前边界框IoU小于阈值的边界框
        order = order[inds + 1]
    return keep

def gesture(runner,size,epoches,thread_name,player):
    name={0: 'four', 1: 'one', 2: 'palm', 3: 'peace', 4: 'three', 5: 'no gesture'}
    print(f"{thread_name},start...")
    d_flag=0
    global video_frame,g_flag,display_frame_gesture
    sum_time=0
    
    for i in range (epoches):
        img = video_frame.copy()
        local_img=load_image_camera(img,size)
        start=time.time()
        output_data = runner.run(local_img.astype(np.float32)/255.0)
        sum_time +=time.time()-start
        detect_score = np.squeeze(output_data) 
        output_data = np.transpose(detect_score)
        # 前4个是矩形框 x, y, width, height
        boxes = output_data[:, :4] 
        print(boxes.shape)
        # 后2个是①的概率，②的概率
        scores = output_data[:, 4:]
        print(scores.shape)
        # 计算每个边界框最高的得分
        max_scores = np.max(scores, axis=1)
        # 找到满足一定准确率的框【修改点在这里】
        keep = max_scores >= 0.2
        # 得到符合条件的边界框和得分
        filtered_boxes, filtered_scores = boxes[keep], scores[keep]
        text='gesture_recognition' 
        display_frame_gesture = img.copy()
        cv2.putText( display_frame_gesture,text,(0,30),cv2.FONT_HERSHEY_COMPLEX,0.8,font_color,1)
        (height, width) = display_frame_gesture.shape[:2]
        # 计算每个边界框的最高得分
        max_scores = np.max(filtered_scores, axis=1)
        # 进行处理
        keep = non_max_suppression(filtered_boxes, max_scores)
        # 最后留下的候选框
        final_boxes = filtered_boxes[keep]
        final_scores=filtered_scores[keep]
        # 目标的索引
        indexs = np.argmax(final_scores, axis=1)
        print(final_scores)
        for i, box in enumerate(final_boxes):
            x,y,w,h = box
            # x,y 是中心点的坐标，而且是占宽高的百分比
            x1,y1 = (x-w/2)*width, (y-h/2)*height
            x2,y2 = (x+w/2)*width, (y+h/2)*height
            color_v = (0, 0, 255) if name[indexs[i]] == 'palm' else (255, 0, 0)
            txt=str(name[indexs[i]])+str(':%.2f %%'% (final_scores[i][indexs[i]]*100))
            cv2.rectangle(display_frame_gesture , (int(x1), int(y1)), (int(x2), int(y2)),  color_v, 2)
            cv2.putText(display_frame_gesture ,txt,(int(x1), int(y1)),cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
            
            if name[indexs[i]] == 'palm' and g_flag ==0:
                player.pause()
                with lock:
                    g_flag = 1
                    cv2.putText( display_frame_gesture,"暂停",(0,50),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)                          
                print("Pause")
            elif name[indexs[i]] == 'three' and g_flag ==1:
                player.resume() 
                with lock:
                    g_flag = 0   
                    cv2.putText( display_frame_gesture,"播放",(0,50),cv2.FONT_HERSHEY_COMPLEX,1.0,font_color,1)   
                print("Play")             

        
        
    print(f"{thread_name},exit...sum infer time : {sum_time},avg time: {sum_time/epoches}")
    
#读取food101的分类列表，为菜品分类提供标签
def read_food101_classes(file_path):
    classes=[]
    with open (file_path,'r') as file:
        for line in file:
            class_name = line.strip().split(' ')
            classes.append(class_name)
    return classes


#读取food101的分类列表，为菜品分类提供标签
def read_gsc_classes(file_path):
    classes=[]
    with open (file_path,'r') as file:
        for line in file:
            class_name = line.strip().split(' ')
            classes.append(class_name)
    return classes

#菜品分类
def food_recognition_infer(runner,size,epoches,thread_name):
    sum_time=0
    print(f"{thread_name},start...")
    file_path='food101/meta/classes.txt'
    classes = read_food101_classes(file_path)

    global video_frame,display_frame_food
    
    
    for i in range (epoches):
        img= video_frame.copy()
        start=time.time()
        local_img=load_image_camera(img,size)
        prediction = runner.run(local_img.astype(np.float32)/255.0)
        sum_time +=time.time()-start
        softmax_output=tf.nn.softmax(prediction[0],axis=-1)
        confidence = tf.reduce_max(softmax_output).numpy()
        class_index =tf.argmax(softmax_output,axis=-1).numpy()
        with lock:
            text='food_recognition'
            class_text=classes[int(class_index)][0]+str(confidence)+'%'
            display_frame_food=img
            cv2.putText( display_frame_food,text,(0,30),cv2.FONT_HERSHEY_COMPLEX,0.8,font_color,1)
            if confidence>0.7:
                cv2.putText( display_frame_food,class_text,(0,60),cv2.FONT_HERSHEY_COMPLEX,0.5,font_color,1)
            
    print(f"{thread_name},exit...sum infer time : {sum_time},avg time: {sum_time/epoches}")

#语音识别
def gsc_infer(runner,test_loader,epoches,thread_name):
    sum_time=0
    print(f"{thread_name},start...")
    classes = ['bed', 'bird', 'cat', 'dog', 'down', 'eight',
               'five', 'four', 'go', 'happy', 'house', 'left',
               'marvin', 'nine', 'no', 'off', 'on', 'one', 'right',
               'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
    global display_frame_gsc
    text="Speech command"
    for i in range (epoches):
        start=time.time()
        sample=test_loader[random.randint(0,99)]
        # 读取测试图
        voice=sample[0].permute(0,2,3,1)
        prediction = runner.run(voice)
        sum_time +=time.time()-start
        return_classes = np.argmax(prediction[0])
        display_frame_gsc = np.ones([img_size, img_size, 3], dtype=np.uint8)
        cv2.putText(display_frame_gsc, text, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, font_color, 1)
        cv2.putText(display_frame_gsc, classes[return_classes], (80, 160), cv2.FONT_HERSHEY_COMPLEX, 1.0, font_color, 1)
    print(f"{thread_name},exit...sum infer time : {sum_time},avg time: {sum_time/epoches}")
    
#加载模型
def cpu_load_model(data_set):
    device = torch.device('cpu')
    if data_set=="gsc":
         model=TFLiteRunner("tflite_model/gsc_0_8.tflite")
    elif data_set=="fall_down_detection":
        model=TFLiteRunner("tflite_model/efficientdet-lite0.tflite")
    elif data_set=="object_detection":
        model=TFLiteRunner("tflite_model/efficientdet-lite0.tflite")
    elif data_set=="food_recognition":
        model=TFLiteRunner("tflite_model/efficient_b0.tflite")
    elif data_set=="gesture_recognition":
        model = TFLiteRunner("tflite_model/yolov8.tflite")
    print(data_set+"模型加载完毕")
    return  model


if __name__ == '__main__':
    infer_epoches=10000
    
    #开始视频捕获线程
    main_thread = threading.Thread(target=capture_video)
    main_thread.start()

    #data_set=["gsc"]
    #data_set=["fall_down_detection"] 
    #data_set=["object_detection"]
    #data_set=["gesture_recognition"]
    #data_set=["food_recognition"]
    #data_set=["gsc","fall_down_detection","object_detection"]
    #data_set=["gsc","fall_down_detection","object_detection","food_recognition"]
    #data_set=["gsc","fall_down_detection","object_detection","gesture_recognition"]
    #选择任务
    data_set=["gsc","fall_down_detection","object_detection","gesture_recognition","food_recognition"]
    #对应任务的数据
    test_loader={}
    #对应任务的运行时间
    time_record={"gsc":[],"fall_down_detection":[],"object_detection":[],"food_recognition":[],"gesture_recognition":[]}
    model={}
    #控制gsc处理数据的batch_size 大小,为1时说明，每次只推理一段语音
    
    infer_batch_size =100
    for i in range(len(data_set)):
        model[data_set[i]]=cpu_load_model( data_set[i])

    
    all_start=time.time()
    
    
    for i in range(len(data_set)):
        if data_set[i] =='gsc':
            test_loader[data_set[i]]=load_data_set(data_set[i],infer_batch_size)

    print("**********start********")
    player= vp.Player()
    #player.add_callback(vlc.EventType.MediaPlayerTimeChanged, my_call_back)
    # 播放本地mp3
    player.play("./test.mp4")
            
    #定义thread1-5线程
    thread1 = threading.Thread(target=gsc_infer, args=(model['gsc'],test_loader['gsc'], infer_epoches, 'Thread 1 gsc'))
    thread2 = threading.Thread(target=fall_down_infer, args=(model['fall_down_detection'],[img_size,img_size], infer_epoches, 'Thread 2_down_detection'))
    thread3 = threading.Thread(target=object_infer, args=(model['object_detection'],[img_size,img_size], infer_epoches, 'Thread 3 object_detection',player))
    thread4 = threading.Thread(target=gesture,args=(model['gesture_recognition'], [640,640], infer_epoches, 'Thread 4 gesture_recognition',player))
    thread5 = threading.Thread(target=food_recognition_infer, args=(model['food_recognition'],[224,224], infer_epoches, 'Thread 5 food_recognition'))
    
    #定义thread1-5线程开始
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    


    #主线程在循环显示画面
    print("play....")
    while True:
        # combined_images_1 = np.concatenate((display_frame ,display_frame,display_frame_fall) ,axis=1)
        # combined_images_2 = np.concatenate((display_frame_object,display_frame_gesture,display_frame_food),axis=1)
        # combined_images=np.concatenate((combined_images_1,combined_images_2),axis=
        combined_images = np.concatenate(
            (cv2.resize(display_frame, (int(img_size*0.8), int(img_size *0.8))),
             cv2.resize(display_frame_fall, (int(img_size *0.8), int(img_size *0.8))),
             cv2.resize(display_frame_object, (int(img_size*0.8), int(img_size *0.8))),
             cv2.resize(display_frame_gesture, (int(img_size *0.8), int(img_size *0.8))),
             cv2.resize(display_frame_food, (int(img_size *0.8), int(img_size *0.8))),
            cv2.resize(display_frame_gsc, (int(img_size * 0.8), int(img_size * 0.8)))),
            axis=1)
        cv2.imshow("Video Capture",combined_images )
        if cv2.waitKey(1) & 0xFF == ord('q') or exit_sign == True:
            break
    cv2.destroyAllWindows()
    print('display done')
    
    
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    
    #thread1-5线程结束后通知main_thread退出
    exit_sign=True
    

    main_thread.join()

    
    end_time=time.time()
    print(f'总执行时间为：{end_time-all_start}')
    exit(0)



