# 모델 개선사항
## 키 아이디어 :
* 손실 함수에서 마지막 선형 계층의 가중치 차이를 이용하여, 논문의 핵심 아이디어인 최소 마진 최대화는 그대로 반영
## <개선 사항>
1. L1/L2 정규화 추가: 가중치를 제어하여 과적합 방지와 희소성 확보
2. MLP 구조 활용: 복잡한 데이터 패턴을 학습할 수 있도록 은닉층 추가
3. Early Stopping 도입: 학습을 조기에 종료하여 불필요한 계산을 방지하고 과적합 방지
4. 학습률 스케줄러: 학습 초반에는 큰 변화, 후반에는 미세 조정을 지원
5. Random Search 알고리즘으로 최적의 하이퍼파라미터로 모델 튜닝
    - Learning Rate : [0.001, 0.0005, 0.0001]
    - Batch Size : [64, 128]
    - Hidden Layer : [512, 1024]
    - Drop Out : [0.3, 0.4]
    - 결과 : {'lr': 0.0005, 'batch_size': 128, 'hidden_dim': 512, 'dropout_rate': 0.4}

## <기존 데이터 셋실험 결과>

| File | Dataset | Paper's Accuracy   | 개선모델 Accuracy |
|-------|--------|---------------|----------|
| M3SVM_upgrade | cornell   |  0.865  | 0.8675(+ 0.0025)     |
| M3SVM_upgrade | ISOLET   |  0.945  | 0.9679(+ 0.0229)        |
| M3SVM_upgrade | HHAR   |  0.981  | 0.9947(+ 0.0137)       |
| M3SVM_upgrade | USPS   |  0.956  | 0.9839(+ 0.0279)        |
| M3SVM_upgrade | ORL   |  0.975  | 0.9875(+ 0.0125)        |
| M3SVM_upgrade | Dermatology   |  0.988  | 1.000(+ 0.012)       |
| M3SVM_upgrade | Vehicle   |  0.800  | 0.9118(+ 0.1118)       |
| M3SVM_upgrade | Glass   |  0.744  | 0.8372(+ 0.0932)       |

## <새로운 분류 데이터셋 실험 결과>

### FashionMNIST 학습 결과

| Model | File   | Test Accuracy |
|-------|--------|---------------|
| SVM + RBF     | fashionMnist_otherModel | 0.8669        |
|  DecisionTree   | fashionMnist_otherModel | 0.7901        |
| RandomForest    | fashionMnist_otherModel | 0.8735        |
| XGBoost    | fashionMnist_otherModel | 0.8926       |
| LightGBM    | fashionMnist_otherModel | 0.89        |
| M3SVN(논문 모델)    | fashionMnist_upgrade | 0.8563        |
| M3SVN(개선점 반영)    | fashionMnist_upgrade | **0.9025**        |
| M3SVN(하이퍼 파라미터 튜닝)    | fashionMnist_upgrade | **0.9114**        |