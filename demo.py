import cv2
import os
import numpy as np
from PIL import Image #pillow
import pyttsx3
import  sys
import json
def makeDir(engine):
    if not os.path.exists("face_trainer"):
        print("创建预训练环境")
        engine.say('检测到第一次启动，未检测到环境，正在创建环境')
        engine.say('正在创建预训练环境')
        os.mkdir("face_trainer")
        engine.say('创建成功')
        engine.runAndWait()
    if not os.path.exists("Facedata"):
        print("创建训练环境")
        engine.say('正在创建训练环境')
        os.mkdir("Facedata")
        engine.say('创建成功')
        engine.runAndWait()
def getFace(cap,face_id):
    # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
    #cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #face_id = input('\n enter user id:')
    print('\n Initializing face capture. Look at the camera and wait ...')
    count = 0
    while True:
        # 从摄像头读取图片
        sucess, img = cap.read()
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
            count += 1
            # 保存图像
            cv2.imwrite("Facedata/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])
            cv2.imshow('image', img)
        # 保持画面的持续。
        k = cv2.waitKey(1)
        if k == 27:   # 通过esc键退出摄像
            break
        elif count >= 100:  # 得到1000个样本后退出摄像
            break

    cv2.destroyAllWindows()
def getImagesAndLabels(path,detector):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # join函数的作用？
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x: x + w])
                ids.append(id)
        return faceSamples, ids
def trainFace():
    # 人脸数据路径
    path = 'Facedata'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print('Training faces. It will take a few seconds. Wait ...')
    faces, ids = getImagesAndLabels(path,detector)
    recognizer.train(faces, np.array(ids))
    recognizer.write(r'face_trainer\trainer.yml')
    print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))
def checkFace(cam,names,engine):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    idnum = 0
    #names = ['zongyong', 'zhangmin', 'shanglanqing']
    #cam = cv2.VideoCapture(0)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                idnum = names[idnum]
                confidence = "{0}%".format(round(100 - confidence))
                say(engine, "欢迎      "+idnum+"签到成功！")
                return
            else:
                idnum = "unknown"
                confidence = "{0}%".format(round(100 - confidence))

            cv2.putText(img, str(idnum), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 0), 1)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10)
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
def say(engine,str):
    engine.say(str)
    engine.runAndWait()

if __name__ == '__main__':
    names = []
    if os.path.exists("name.txt"):
        with open("name.txt") as f:
            names = json.loads(f.read())
            print(names)
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20)
    makeDir(engine)
    while True:
        say(engine, "是否要录入新的人脸信息      ")
        say(engine, "输入0 代表是 输入其他代表不是")
        value = input("0：是 or other：否")
        if value == '0':
            say(engine, "请输入您的姓名，注意要写成拼音形式")
            name = input("请输入姓名：")
            names.append(name)
            say(engine,"正在打开摄像头")
            cam = cv2.VideoCapture(0)
            say(engine, "注视摄像头，开始采集人脸数据")
            getFace(cam,len(names)-1)
            say(engine, "采集完毕，开始训练")
            trainFace()
            say(engine, "训练完毕，开始人脸识别 ，按esc键将会终止本次识别")
        else:
            say(engine, "开始人脸识别")
            say(engine, "正在打开摄像头")
            cam = cv2.VideoCapture(0)

        checkFace(cam,names,engine)
        say(engine, "继续输入 0 退出系统 ，输入 其他任意键 录入新的人脸")
        key = input("输入key：（0 - 退出系统 ，other - 重新启动系统）")
        if key == '0':
            #将姓名保存到文件
            with open("name.txt",'w') as f:
                f.write(json.dumps(names))
            say(engine, "信息已保存")
            say(engine, "再见")
            sys.exit(0)
