# 논문제목 :  A study on the prediction algorithm of diabetes based on XGBoost: 
 Data from the 2016~ 2018 Korea National Health and Nutrition Examination Survey

    - 목적 : 당뇨병 조기 예측을 위한 머신러닝 기반 모델 연구
    - 주요기법 : SMOTE와 XGBoost 활용
    - 선정이유 : 본 연구의 기초 머신러닝 학습에 활용
    - 링크 : https://ca.skku.edu:8443/link.n2s?url=https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10547856
    - 코드 : https://colab.research.google.com/drive1mZZpJGfq7OJD3st0I0LzhL7k2RFlIJwM#scrollTo=nJ86TuLcY0_6

21년 한국통신학회 동계 학술종합발표회 투고 논문(17E-54) "XGBoost 기반 당뇨병 예측 알고리즘 연구:국민건강영양조사 2016~2018을 이용하여"의 Open Source Repository 입니다.(ISSN:2383-8302(Online) Vol.74)본 레포는 '다양한 관점에서의 보건영양데이터'를 주제로, 제7기 2018 국민 보건영양조사의 데이터를 이용했습니다.본 연구에서의 핵심 목표는 **당뇨병 예측모델 개발** 이며 본 Repository에서는 프로젝트의 아키텍쳐, 코드, 결과 그래프 등이 포함되어 있습니다.연구 전반의 기록과 과정을 노션 페이지에서도 확인할 수 있습니다.

https://www.notion.so/hobbeskim/XGBoost-20-2-2740a0d75839481b8cbefa7cdab69466
## Architecture
![Archi](https://user-images.githubusercontent.com/57410044/106548637-b39bfc00-6552-11eb-91ce-3b629b599dfc.png)

## Model Evaluation
![roc](https://user-images.githubusercontent.com/57410044/106548802-070e4a00-6553-11eb-92ba-c49ce1859fd2.jpg)
![values](https://user-images.githubusercontent.com/57410044/106548809-0a093a80-6553-11eb-9e4c-09cdf0572662.png)


## Files Description

- ./Data : 실제로 연구에 이용한 데이터 파일, 국민건강영양조사 데이터 가공
- ./Data EDA : 데이터를 분석하고 시각화 한 과정의 자료가 존재
- ./Data Processing : 모델 학습과 예측을 위해 전처리를 수행한 과정의 파일 존재 
- ./Etc : 기타 개발과정의 오류나 산출물 등
- ./Results : 결과 그래프 원본 이미지
- FOLD_EVALUATION.ipynb : 가공된 데이터 -> 최종 결과 수행 시퀀스


### 논문재현
 0 - FOLD START
ACC :  0.8673894912427023
Precision :  0.8034682080924855
recall(TP rate) :  0.7533875338753387
F1 :  0.7776223776223776
ROC SCORE :  0.8357299115159826

 1 - FOLD START
ACC :  0.8457047539616347
Precision :  0.775623268698061
recall(TP rate) :  0.7291666666666666
F1 :  0.7516778523489933
ROC SCORE :  0.814890081799591

 2 - FOLD START
ACC :  0.8707256046705588
Precision :  0.7824933687002652
recall(TP rate) :  0.8016304347826086
F1 :  0.7919463087248321
ROC SCORE :  0.8514770705802333

 3 - FOLD START
ACC :  0.8423686405337781
Precision :  0.7493472584856397
...
Precision :  0.8628571428571429
recall(TP rate) :  0.7330097087378641
F1 :  0.7926509186351706
ROC SCORE :  0.8360093016370388