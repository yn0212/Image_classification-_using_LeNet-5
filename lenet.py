import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout

#정확도 시각화
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import font_manager
#전처리 없이 신경망 구축
#케라스에서 제공하는 Sequential API를 사용
#모델 계층을 순차적으로 쌓아 올려 LeNet-5라는 클래스 만들기
#입력 값은 이미지이며, 출력 값은 클래스의 확률 벡터
 
num_classes = 10
class LeNet(Sequential):
    #생성자
    def __init__(self, input_shape, nb_classes): 
        super().__init__()
#Conv2D(필터 개수,커널의 행과 열,필터를 적용하는 간격,
# ,렐루 활성화 함수 사용,입력이미지형태,패딩값same=입출력이미지크기같음)
        self.add(Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='relu',
                        input_shape=input_shape, padding="same")) 
# AveragePooling2D(연산 범위,계산 과정에서 한 스텝마다 이동하는 크기
# ,'valid' 값은 이미지크기유지비활성화)                        
        self.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid')) 
 
        self.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='relu',
                        padding='valid'))
        self.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        #완전 연결층
        self.add(Flatten()) #완전연결층 가기전  Flatten 이용해 1차원데이터로 변형
        #입력층과 출력층을 연결하기위한 밀집층Dense사용 
        self.add(Dense(120, activation='relu'))
        self.add(Dense(84, activation='relu'))
         # 출력층의 노드 개수는 2이고 소프트맥스 활성화 함수가 적용된 출력층   
        self.add(Dense(nb_classes, activation='softmax'))
        #학습 방식 설정
#optimizer : 
#손실함수를 사용해 구한 값으로 기울기를 구하고 신경망의 파라미터를 학습에 어떻게 반영할지 결정하는 방법
#loss : 최적화 과정에서 사용될 손실함수 설정.
# 다수의 클래스 사용하므로loss='sparse_categorical_crossentropy'
#metrics:모델의 평가기준 지정
        self.compile(optimizer='adam',
                     loss=categorical_crossentropy,
                     metrics=['accuracy'])
 
# 클래스(LeNet)를 호출하여 LeNet-5라는 모델을 생성
#출력은 개와 고양이 표현하는 2
model = LeNet((100,100,3), num_classes) 
model.summary()
 
#필요한 파라미터에 대한 값들을 초기화, 개와 고양이 이미지를 호출
EPOCHS = 100 #원래100~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BATCH_SIZE = 100
image_height = 100
image_width = 100
train_dir = "./data/num/train/"
valid_dir = "./data/num/test1/"
 
#ImageDataGenerator를 사용하여 이미지에 대한 전처리
train = ImageDataGenerator(
#1/255로 스케일링하여 0~1 범위로 변환
                rescale=1./255,
#이미지 회전 범위 10은 0~10도 범위 내 임의로 회전
                rotation_range=10,
#그림을 수평으로 랜덤하게 평행 이동시키는 범위
                width_shift_range=0.1,
#그림을 수직으로 랜덤하게 평행 이동시키는 범위
                height_shift_range=0.1,
)
 
#flow_from_directory 메서드:
#   폴더 구조를 그대로 가져와서 ImageDataGenerator에 실제 데이터를 채워 줌
train_generator = train.flow_from_directory(train_dir,#훈련 이미지 경로
 ##이미지크기,모든 이미지는 이 크기로 자동 조정
                                            target_size=(image_height, image_width),
                                 #그레이스케일이면 'grayscale', 색상 'rgb' 사용
                                            color_mode="rgb",
          #배치당 generator에서 생성할 이미지 개수                                   
                                            batch_size=BATCH_SIZE,
          #이미지를 임의로 섞기 위한 랜덤한 숫자
                                            seed=1,
         #이미지 섞어서 사용하려면 shuffle을 True
                                            shuffle=True,
#예측할 클래스가 두 개뿐이라면 "binary"를 선택,아니면 "categorical"을 선택                                      
                                            class_mode="categorical")
# ImageDataGenerator 사용시 파라미터 설정을 사용해 데이터 전처리 쉽게 할 수 있음.
valid = ImageDataGenerator(rescale=1.0/255.0)
#flow_from_directory 메서드:  폴더 구조를 그대로 가져와서
# ImageDataGenerator에 실제 데이터를 채워 줌
valid_generator = valid.flow_from_directory(valid_dir,
                                            target_size=(image_height, image_width),
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            seed=7,
                                            shuffle=True,
                                            class_mode="categorical")
train_num = train_generator.samples
valid_num = valid_generator.samples
 
# tf.keras.callbacks.TensorBoard 콜백을 추가하여 로그가 생성되고 저장됨
#callback은 에포크의 시작과 끝처럼 이벤트가 발생할 때 호출됨
#- 정확도가 특정 임계치를 초과할 때 저장
#- 이메일을 보내거나 학습을 종료할 때 알림 보내기
log_dir = "./log/"#로그 파일이 기록될 위치
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,#로그 파일이 기록될 위치
                         #매 에포크마다 출력을 히스토그램으로 기록
                                             histogram_freq=1,
         #훈련이 진행시간 및CPU 등에 대한 사용을 관리,두 번째 배치부터 계산
                                              profile_batch=0) 
#모델 학습
 
history=model.fit(train_generator,#입력 데이터
          epochs=EPOCHS,#학습 횟수
          #한 에포크에서 사용한 스텝
          steps_per_epoch=train_num // BATCH_SIZE,
          #성능을 모니터링하는 데 사용하는 데이터셋을 설정
          validation_data=valid_generator,
          #한 에포크가 종료될 때 사용되는 검증 스텝 개수
          validation_steps=valid_num // BATCH_SIZE,
          #텐서보드라는 콜백 함수를 생성후 파라미터 넣기@
          callbacks=[tensorboard_callback],
          #훈련의 진행 과정을 보여 주기설정
          verbose=1) 


font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()
 
plt.rcParams["font.family"] = font_family
#accuracy: 매 에포크에 대한 훈련의 정확도
accuracy = history.history['accuracy'] 
#val_accuracy: 매 에포크에 대한 검증의 정확도
val_accuracy = history.history['val_accuracy']
#loss: 매 에포크에 대한 훈련의 손실 값
loss = history.history['loss']
# val_loss: 매 에포크에 대한 검증의 손실 값
val_loss = history.history['val_loss']
 
epochs = range(len(accuracy))
 
plt.plot(epochs, accuracy, label="훈련 데이터셋")
plt.plot(epochs, val_accuracy, label="검증 데이터셋")
plt.legend()
plt.title('정확도')
plt.figure()
 
plt.plot(epochs, loss, label="훈련 데이터셋")
plt.plot(epochs, val_loss, label="검증 데이터셋")
plt.legend()
plt.title('오차')

model.save('yn_lenet_model.h5')

#predict_classes() 메서드를 사용하여 결과를 예측
class_names = ['0','1','2','3','4','5','6','7','8','9']
validation, label_batch = next(iter(valid_generator)) #validation: 이미지 배열, label_batch : 정답 레이블 배열

prediction_values = model.predict(validation) #예측값
prediction_values = np.argmax(prediction_values, axis=1) #이미지 배열의 예측값
 
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
for i in range(100):
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
    print('prediction_values[i]=',prediction_values[i])
    print('np.argmax(label_batch[i])=',np.argmax(label_batch[i]))
    if prediction_values[i] == np.argmax(label_batch[i]): #이미지배열 예측값==정답 레이블배열의 최대값인덱스
        ax.text(3, 17, class_names[prediction_values[i]], color='yellow', fontsize=14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)
plt.show()
