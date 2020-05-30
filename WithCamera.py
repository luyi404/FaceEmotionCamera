import cv2
from Train import *
from PIL import Image

model = torch.load('./model//model29.pth').cuda()  #加载已经训练好的pytorch模型

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((48,48)),
    torchvision.transforms.ToTensor()
])#定义图像变换以符合网络输入

emotion = ["angry","disgust","fear","happy","sad","surprised","neutral"]#表情标签
emotion_ch = ["生气", "反感", "害怕", "开心", "伤心", "惊讶", "司马脸"]
#可惜了好像不支持中文

cap = cv2.VideoCapture(0)# 摄像头，0是笔记本自带摄像头
face_cascade = cv2.CascadeClassifier("F:\PY\envs\pytorch_gpu\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
#opencv自带的一个面部识别分类器
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
    frame= frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.1,3)
    img = frame
    if(len(face)>=1):
        (x,y,w,h)= face[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img = frame[:][y:y+h,x:x+w]
    # 如果分类器能捕捉到人脸，就对其进行剪裁送入网络，否则就将整张图片送入
    img = Image.fromarray(img)
    img = transforms(img)
    img = img.reshape(1,1,48,48).cuda()
    pre = model(img).max(1)[1].item()   #获取表情的值
    frame = cv2.putText(frame, emotion[pre], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
    #显示窗口第一个参数是窗口名，第二个参数是内容
    cv2.imshow('emotion', frame)
    if cv2.waitKey(1) == ord('q'):#按q退出
        break
cap.release()
cv2.destroyAllWindows()