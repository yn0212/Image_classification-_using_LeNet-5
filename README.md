# Image_classification-_using_LeNet-5 :computer:  :pencil2:
![Footer](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=footer)

# :pushpin:Project Description
LeNet-5 신경망을 사용해 로컬데이터의 필기체를 96%확률로 인식하는 프로그램이다.
-----------------------------------------------------------

# :pushpin:Project Purpose
- 이미지 분류를 위한 신경망 모델 LeNet-5를 사용해 필기체 인식률을 95%이상 높이기
- 신경망 구조 생성부터 모델 학습, 예측까지의 과정을 구현해보고 적절한 파라미터 값을 찾으며 공부한 내용 복습

# :pushpin:Device used for the project
- 학습에 사용한 GPU : GTX1660 Supper

# :pushpin:Project Dataset
- 사용한 데이터셋 : 직접 만든 로컬 데이터
- 클래스 수 : 0~9 총 10개(훈련 데이터 : 약 400-500개 , 검증 데이터: 약 60개)

# :pushpin:Project Results
![afe213166d1a6bbb2d44465a52d8d97c94960912_re_1673242484453](https://user-images.githubusercontent.com/105347300/211253351-b02c9ef4-9275-419d-a556-a26afc951867.png)

- 필기체 인식률 : 96%

![3c470c182438863e69357c119c0ef5bc32ed6de1_re_1673242484453](https://user-images.githubusercontent.com/105347300/211253417-19712100-0453-4bf5-a87d-d97a7f9b03e9.png)

- 훈련데이터 정확도: 0.9932
- 훈련데이터 오차:0.0191
- 검증데이터 정확도:0.96
- 검증데이터 오차:0.2418

# :pushpin:Project Process

## :loudspeaker:정의
- LeNet-5는 합성곱(convolutional)과 다운 샘플링(sub-sampling)(혹은 풀링)을 반복적으로 거치면서 마지막에 완전연결층에서 분류를 수행하는 신경망
-LeNet 구조

![e1390f5d49957a0115066c058efdec89f3c3189c_re_1673242484455](https://user-images.githubusercontent.com/105347300/211254579-e2d5ef45-7872-4d00-941a-ee69929af675.png)

- 간단한 신경망이라 예제로 구현 가능 할 것 같아 선택 !

## :loudspeaker:코드로 구현할 신경망
![1d2872f4150b64fd713390eb4afe41ac49ed2889_re_1673242484454](https://user-images.githubusercontent.com/105347300/211254676-b6d17385-bd8c-4135-945a-48eb9fecd425.png)

- 32×32 크기의 이미지에 합성곱층과 평균 풀링층이 쌍으로 두 번 적용된 후 완전연결층을 거쳐 이미지가 분류되는 신경망

![1e2ac07166224969c4e7882ba6b3e190e2b39357_re_1673242484454](https://user-images.githubusercontent.com/105347300/211254746-9cf88006-65e0-4ed0-993b-86a6358d9798.png)

## :loudspeaker:과정

### :bulb: 모델 계층을 순차적으로 쌓아 올려 LeNet클래스 만들기


    num_classes = 10
    class LeNet(Sequential):
    def __init__(self, input_shape, nb_classes): 
        super().__init__()
        #Conv2D(필터 개수,커널의 행과 열,필터를 적용하는 간격,
        # ,렐루 활성화 함수 사용,입력이미지형태,패딩값same=입출력이미지크기같음)
        self.add(Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='relu',input_shape=input_shape, padding="same")) 
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


- 위의 표처럼 신경망 구성함

### :bulb: 모델 생성 , 전처리
- ImageDataGenerator - 객체 생성==> 내 데이터셋의 양이 적으므로 전처리 과정을 통해 이미지를 증식함.
- flow_from_directory-객체 생성 ==> 폴더 구조를 그대로 가져와서 imageDataGenerator에 실제 데이터를 채워줌.(자신의 로컬 폴더에있는 데이터를 사용하기 위해 생성함.)

        # 클래스(LeNet)를 호출하여 LeNet-5라는 모델을 생성
        model = LeNet((100,100,3), num_classes) 
        model.summary()
        
### :bulb: 모델 학습
- epoch=50일때

![image](https://user-images.githubusercontent.com/105347300/211257076-478e1c50-df75-45dd-a5da-80622c6ee2da.png)

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

### :bulb: 결과 예측
- 검증데이터 100개를 랜덤으로 가져와 예측이 틀리면 빨간색, 맞으면 노란색 글씨로 출력
- 시행착오 결과 epoch를 충분히 적절히 설정해야함
- 최적의 epochs =100으로 설정함.

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
            
### :bulb: 예측 결과 , 모델 성능
- 모델 파일 이름 : yn_lenet_model.h5 
![afe213166d1a6bbb2d44465a52d8d97c94960912_re_1673242484453](https://user-images.githubusercontent.com/105347300/211256338-6ab2f5c0-8555-4452-8c57-f9073af9b3b7.png)
- epochs =  100 으로 설정
- 100개의 검증이미지 중  96 개 맞춤 
- 필기체 인식률 : 96%
- 훈련데이터 정확도: 0.9932
- 훈련데이터 오차:0.0191
- 검증데이터 정확도:0.96
- 검증데이터 오차:0.2418

