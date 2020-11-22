# import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv
import time                     #計時用
# import threading                #讓Python多工的內建插件

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# class face_emotion():
#     def __init__(self):

#         # 使用特征提取器get_frontal_face_detector
#         self.detector = dlib.get_frontal_face_detector()
#         # dlib的68点模型，使用作者训练好的特征预测器
#         self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#         #嘴巴張開的臨界值
#         self.mouthOpen_Criterion = 0.19
#         self.mouthOpen_Init = 0.35
#         #眼睛閉上的臨界點
#         self.eyesClose_Criterion = 0.25
#         self.eyesClose_Init = 0.4

#     def calc_concerntration(self, mouth_openRate, eyes_openRate):
#         degree = (((self.mouthOpen_Init - mouth_openRate) / self.mouthOpen_Init) + (eyes_openRate / self.eyesClose_Init)) / 2
#         return degree

#     def learning_face(self, _input):

#         # 把圖片縮小一半
#         #capWidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#         #capHeight = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         #im_rd = cv2.resize(im_rd, ((int)(capWidth/2), (int)(capHeight/2)))

#         imageInput = cv2.imdecode(np.fromstring(_input.read(), np.uint8), cv2.IMREAD_UNCHANGED)
#         # 取灰度
#         img_gray = cv2.cvtColor(imageInput, cv2.COLOR_RGB2GRAY)

#         # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
#         faces = self.detector(img_gray, 0)

#         # 待会要显示在屏幕上的字体
#         font = cv2.FONT_HERSHEY_SIMPLEX

#         # 如果检测到人脸
#         if(len(faces)!=0):

#             # 对每个人脸都标出68个特征点
#             for i in range(len(faces)):
#                 # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
#                 for k, d in enumerate(faces):
#                     # 用红色矩形框出人脸
#                     cv2.rectangle(imageInput, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
#                     # 计算人脸热别框边长
#                     self.face_width = d.right() - d.left()

#                     # 使用预测器得到68点数据的坐标
#                     shape = self.predictor(imageInput, d)
#                     # 圆圈显示每个特征点
#                     for i in range(68):
#                         cv2.circle(imageInput, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
#                         #cv2.putText(im_rd, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255))


#                     # 眼睛睁开程度
#                     eye_hight = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
#                                 shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y) / 4
#                     eye_width = (shape.part(39).x - shape.part(36).x + shape.part(45).x - shape.part(42).x) / 2
#                     eyes_openRate = eye_hight / eye_width
#                     cv2.putText(imageInput, 'eyes_openRate  %.2f  limit %.2f' %(eyes_openRate, self.eyesClose_Criterion), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 255), 1, 4)

#                     # 嘴巴睜開比例
#                     mouth_width = (shape.part(64).x - shape.part(60).x) / self.face_width  # 嘴巴咧开程度
#                     mouth_higth = (shape.part(66).y - shape.part(62).y) / self.face_width  # 嘴巴张开程度
#                     mouth_openRate = mouth_higth / mouth_width
#                     cv2.putText(imageInput, 'mouth_openRate %.2f  limit %.2f' %(mouth_openRate, self.mouthOpen_Criterion), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 255), 1, 4)

#                     if eyes_openRate <= self.eyesClose_Criterion:
#                         cv2.putText(imageInput, 'eyes close', (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 255), 1, 4)     
#                     if mouth_openRate >= self.mouthOpen_Criterion:
#                         cv2.putText(imageInput, 'mouth open', (350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 255), 1, 4)

#                     concern_degree = self.calc_concerntration(mouth_openRate, eyes_openRate)
#                     print("mouth_openRate:" + str(mouth_openRate))
#                     print("eyes_openRate:" + str(eyes_openRate))

#                     return (concern_degree)

#         else:
#             # 没有检测到人脸
#             cv2.putText(imageInput, "No Face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

#             return ("No Face")
            
#         #窗口显示
#         cv2.imshow("imageInput", imageInput)

#         # # 删除建立的窗口
#         # cv2.destroyAllWindows()  
# faceAI = face_emotion()


app = Flask(__name__)
CORS(app)

@app.route('/')
def start():
    return "start"

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        # concern_degree = faceAI.learning_face(file)
        return str(concern_degree)
    

if __name__ == '__main__':
    app.run_server(
        debug=True,
    )