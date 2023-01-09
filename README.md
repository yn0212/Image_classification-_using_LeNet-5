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



