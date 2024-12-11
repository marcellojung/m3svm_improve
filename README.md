## 기계학습 특론 5조 Final Project Proposal

### 프로젝트 개요
 - 구성원 : 이승필,정성근,임재이,홍나현,김영주
 - 데이터셋 :Fashion-NMIST 등
 - 목적 : M3SVM 모델을 개선하고, 기존 데이터셋에서 개선된 성능을 보이고, 새로운 데이터셋(Fashion-MNIST)에 적용하여 개선된 성능을 보인다.

### git 프로젝트 구조
 - 모델재현 : 논문에 나온 성능 그래도 재현한 결과입니다 논문리스트 들에 나온 코드들을 구현하였습니다.
 - 모델개선 : 논문에 나온 모델 중 M3SVM 모델을 개선한 결과 코드입니다. 기존 논문에서 사용된 데이터셋에서 개선된 결과들을 얻었으며, 이 모델을 기반으로
 새로운 데이터셋인 FashionMNIST 데이터셋에 적용한 결과입니다.

   (데이터셋의 경우 https://www.kaggle.com/datasets/zalando-research/fashionmnist 에서 다운로드 받았습니다.)


### 논문 리스트
 - A study on the prediction algorithm of diabetes based on XGBoost: 
 Data from the 2016~ 2018 Korea National Health and Nutrition Examination Survey
    - 목적 : 당뇨병 조기 예측을 위한 머신러닝 기반 모델 연구
    - 주요기법 : SMOTE와 XGBoost 활용
    - 선정이유 : 본 연구의 기초 머신러닝 학습에 활용
    - 링크 : https://ca.skku.edu:8443/link.n2s?url=https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10547856
    
    - 코드 : https://colab.research.google.com/drive1mZZpJGfq7OJD3st0I0LzhL7k2RFlIJwM#scrollTo=nJ86TuLcY0_6

 - A Comparative Study for Time-to-Event Analysis and Survival Prediction for Heart Failure Condition using Machine Learning Techniques
    - 목적 : 머신러닝 기법을 활용한 심부전 환자의 생존 예측 및 주요 위험 요인 분석
    - 주요기법 : SVM, Decision Tree, Random Forest, XGBoost, LightGBM  활용
    - 선정이유 : 본 연구의 다양한 머신러닝 기법 학습에 활용
    - 링크 : https://plos.figshare.com/articles/dataset/Survival_analysis_of_heart_failure_patients_A_case_study/5227684/1?file=8937223
    
    - 코드 : https://github.com/sauravmishra1710/Heart-Failure-Condition-And-Survival-Analysis

 - Quadratic Multiform Separation: A New Classification Model in Machine Learning
    - 목적 : 이차 다항식 기반 다중 클래스 분류 기법 제안
    - 주요기법 : QMS, LEM  제안 
    - 선정이유 : 응용 분류 모델 사례 중 하나로 비선형적인 분포를 선형적으로 분리하는 새로운  
    - 링크 : https://arxiv.org/abs/2110.04925
    - 코드 : https://colab.research.google.com/drive/15i7OOQhBEMcfxREOP3fypKDos4dhUa_F#scrollTo=dX5_2c87i1Vu
    
  - (성능 개선)Multi-Class Support Vector Machine with Maximizing Minimum Margin
    - 목적 : 다중 클래스 분류에서 최소 마진을 최대화하는 SVM 연구
    - 주요기법 : 다중 목적 최적화 기반의 새로운 SVM 설계
    - 선정이유 : 응용 분류 모델 구현 및 개선을 위한 핵심 논문
    - 링크 : https://arxiv.org/html/2312.06578v2
    - 코드 : https://github.com/zz-haooo/m3svm
### 알고리즘 성능 개선 아이디어 
 - 알고리즘 성능 개선
    - MLP 도입 : 비선형 데이터 처리의 한계 개선
    
        MLP를 추가하여 비선형 데이터를 효과적으로 학습하여 비선형 패턴과 관계를 더 잘 반영하여 모델의 성능을 향상시킴.
        
    - 정규화 항의 개선 : 가중치 벡터의 크기 제어 부족, 확장성 부족 개선

        L1 정규화는 불필요한 가중치를 0으로 만들어 모델을 단순화하고, L2 정규화는 가중치 크기를 제한해 과적합을 방지하며, 두 정규화의 결합으로 일반화 성능을 최적화
      
       모든 클래스 쌍의 마진을 계산하지 않고, 가장 가까운 클래스 쌍의 최소 마진을 강화하는 정규화 항을 설계해 계산량을 줄이고 성능을 유지
        
    - 학습 안정성과 효율성 향상 : 학습 안정성과 효율성 부족 개선
        
        학습률 스케줄러를 도입하여 기존 Adam의 자동 학습률 조정을 보완하여, 학습 후반부 성능을 향상시킴
    - 최적 하이퍼 파라미터 서치 : RandomSearch CV
    
      RandomSearch Cross Validation을 통해 하이퍼파라미터를 효율적으로 탐색.  GridSearch 대비 더 넓은 파라미터 공간을 짧은 시간 내에 탐색 가능.

 - 새로운 데이터셋으로 평가
    - Fashion-MNIST 데이터 : [3번째 논문에서의 사용된 데이터](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?metric=Accuracy)
    - FashionMNIST 데이터셋은 28X28 그레이스케일 이미지이며 7만개의 데이터(테스트 데이터셋 1만) 및 10개의 클래스로 구성되어 있음 

        | 클래스 번호 | 클래스 이름        | 샘플 수 |
        |-------------|--------------------|---------|
        | 0           | T-shirt/top       | 6,000   |
        | 1           | Trouser           | 6,000   |
        | 2           | Pullover          | 6,000   |
        | 3           | Dress             | 6,000   |
        | 4           | Coat              | 6,000   |
        | 5           | Sandal            | 6,000   |
        | 6           | Shirt             | 6,000   |
        | 7           | Sneaker           | 6,000   |
        | 8           | Bag               | 6,000   |
        | 9           | Ankle boot        | 6,000   |

