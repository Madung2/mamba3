# IMU 기반 동작 전이 구간 탐지 실험 계획서

## 1. 연구 주제

**선택적 상태공간 모델을 활용한 IMU 기반 동작 전이 구간 탐지 연구**

본 연구는 IMU(Inertial Measurement Unit) 시계열 데이터를 이용하여 사용자의 정적 동작 상태뿐 아니라, 동작이 변화하는 **전이 구간(transition interval)** 을 탐지하는 것을 목표로 한다.

기존 IMU 기반 Human Activity Recognition(HAR)은 주로 `walking`, `sitting`, `standing`, `lying`과 같은 현재 동작 클래스를 분류하는 문제로 접근되어 왔다. 그러나 실제 응용 환경에서는 단순히 현재 동작을 아는 것보다, 사용자가 **일어나려는지, 앉으려는지, 균형을 잃고 있는지, 자세가 급격히 변하고 있는지**를 빠르게 감지하는 것이 더 중요하다.

특히 보조 로봇, 재활 로봇, 고령자 케어, 낙상 감지, 웨어러블 안전 시스템에서는 동작 클래스 자체보다 **상태 변화의 시점과 방향성**이 핵심 정보가 된다.

---

## 2. 연구 목표

본 연구의 목표는 다음과 같다.

1. IMU 기반 동작 인식을 정적 행동 분류에서 **동작 전이 구간 탐지 문제**로 확장한다.
2. 동작 전이 구간 탐지를 위해 Mamba-3 기반 선택적 상태공간 모델을 적용한다.
3. 기존 시계열 모델인 1D-CNN, GRU, TCN, Transformer Encoder와 비교하여 전이 탐지 성능을 분석한다.
4. 전체 Accuracy뿐 아니라 Transition Precision, Transition Recall, Transition F1, 추론 시간 등을 함께 평가한다.
5. 가속도계와 자이로스코프 신호의 기여도를 비교하여, 회전성 신호가 동작 전이 탐지에 미치는 영향을 분석한다.

---

## 3. 핵심 연구 질문

### RQ1. 동작 전이 구간 탐지는 기존 HAR 분류와 다른 문제인가?

기존 HAR는 주어진 window가 어떤 동작 클래스에 속하는지를 예측한다. 그러나 동작 전이 구간은 하나의 window 안에 여러 동작 특성이 혼재할 수 있으며, 전이 시점이 짧고 불안정하다. 따라서 기존 activity classification 성능만으로는 실제 전이 탐지 능력을 충분히 평가하기 어렵다.

### RQ2. Mamba-3 기반 선택적 상태공간 모델은 전이 구간 탐지에 유리한가?

동작 전이 구간은 이전 상태가 유지되다가 특정 시점에서 급격하게 변화하는 형태를 가진다. 선택적 상태공간 모델은 입력 시퀀스에 따라 상태를 선택적으로 갱신하고 유지할 수 있으므로, IMU 신호에서 상태 변화가 발생하는 구간을 추적하는 데 적합한 구조적 가능성을 가진다.

### RQ3. 자이로스코프 기반 회전성 신호는 동작 전이 탐지에 얼마나 기여하는가?

일어서기, 앉기, 눕기, 방향 전환과 같은 전이 동작은 단순한 선형 가속도뿐 아니라 각속도 변화와도 관련이 있다. 따라서 가속도계만 사용한 경우, 자이로스코프만 사용한 경우, 두 센서를 함께 사용한 경우를 비교하여 회전성 신호의 기여도를 분석한다.

---

## 4. 데이터셋

### 4.1 Main Dataset

**UCI Smartphone-Based Recognition of Human Activities and Postural Transitions**

이 데이터셋은 스마트폰의 IMU 센서를 이용하여 기본 동작과 자세 전이 동작을 수집한 데이터셋이다.

주요 특징은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 피험자 수 | 30명 |
| 센서 위치 | 허리 부근 스마트폰 |
| 센서 종류 | 3축 가속도계, 3축 자이로스코프 |
| 샘플링 주파수 | 50Hz |
| 기본 window | 2.56초 |
| window size | 128 timestep |
| overlap | 50% |
| 입력 채널 | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z |
| 포함 라벨 | 기본 동작 + 자세 전이 동작 |

### 4.2 주요 라벨

#### 기본 동작 라벨

- walking
- walking upstairs
- walking downstairs
- sitting
- standing
- lying

#### 자세 전이 라벨

- stand-to-sit
- sit-to-stand
- sit-to-lie
- lie-to-sit
- stand-to-lie
- lie-to-stand

---

## 5. 실험 문제 정의

본 연구에서는 실험을 두 단계로 구성한다.

---

### 5.1 실험 A: 기존 HAR Activity Classification

기존 Human Activity Recognition 설정과 동일하게, 각 window의 동작 클래스를 분류한다.

#### 입력

```text
X = [128 timestep, 6 channels]
channels = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
```

#### 출력

```text
y = activity class
```

예시 클래스는 다음과 같다.

```text
walking
walking upstairs
walking downstairs
sitting
standing
lying
stand-to-sit
sit-to-stand
sit-to-lie
lie-to-sit
stand-to-lie
lie-to-stand
```

#### 목적

이 실험은 기본 baseline 성격을 가진다.  
Mamba-3가 단순 activity classification에서 기존 모델보다 압도적으로 높아야 하는 것은 아니다. 핵심은 실험 B의 transition detection이다.

---

### 5.2 실험 B: Binary Transition Detection

본 연구의 핵심 실험이다.

기본 동작 라벨은 `non-transition`, 자세 전이 라벨은 `transition`으로 재정의하여 binary classification을 수행한다.

#### 라벨 재정의

| 원래 라벨 | 재정의 라벨 |
|---|---|
| walking | non-transition |
| walking upstairs | non-transition |
| walking downstairs | non-transition |
| sitting | non-transition |
| standing | non-transition |
| lying | non-transition |
| stand-to-sit | transition |
| sit-to-stand | transition |
| sit-to-lie | transition |
| lie-to-sit | transition |
| stand-to-lie | transition |
| lie-to-stand | transition |

#### 입력

```text
X = [128 timestep, 6 channels]
```

#### 출력

```text
y = transition / non-transition
```

#### 목적

이 실험은 모델이 동작 클래스 자체가 아니라, 상태가 바뀌는 구간을 얼마나 잘 탐지하는지 평가한다.

---

## 6. 비교 모델

2페이지 데모용 소논문에서는 baseline을 너무 많이 넣기보다, 대표적인 시계열 모델을 중심으로 비교한다.

| 모델 | 역할 |
|---|---|
| 1D-CNN | 가장 단순한 지역 패턴 기반 baseline |
| GRU 또는 BiLSTM | 전통적인 순차 상태 모델 baseline |
| TCN | dilated convolution 기반 강한 시계열 baseline |
| Transformer Encoder | self-attention 기반 전역 문맥 baseline |
| Mamba-3-based SSM | 제안 모델 |

---

## 7. 제안 모델 개요

### 7.1 입력 구성

모든 모델의 입력은 동일하게 구성한다.

```text
Input shape = [batch_size, 128, 6]
```

각 channel은 다음과 같다.

```text
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

---

### 7.2 Mamba-3 기반 모델 구조

제안 모델은 IMU 시계열을 입력으로 받아 Mamba-3 기반 선택적 상태공간 encoder를 통과시킨 뒤, classification head를 통해 전이 여부를 예측한다.

```text
IMU sequence
    ↓
Linear Projection
    ↓
Mamba-3 / Selective SSM Encoder
    ↓
Temporal Pooling or Last State
    ↓
MLP Classification Head
    ↓
transition / non-transition
```

---

### 7.3 모델 설계 의도

Mamba-3 기반 선택적 상태공간 모델을 사용하는 이유는 다음과 같다.

1. IMU 신호는 연속적인 상태 변화를 포함하는 다변량 시계열이다.
2. 동작 전이 구간은 이전 상태가 유지되다가 특정 시점에서 변화하는 상태 추적 문제로 볼 수 있다.
3. 선택적 상태공간 모델은 입력에 따라 중요한 정보를 유지하거나 갱신할 수 있다.
4. Transformer보다 긴 시퀀스 처리에서 효율적인 구조를 기대할 수 있다.
5. 자이로스코프 기반 회전성 신호와 상태 변화의 관계를 분석할 수 있다.

---

## 8. 평가 지표

Accuracy만으로는 전이 탐지 성능을 충분히 평가하기 어렵다.  
전이 구간은 일반 동작 구간보다 적을 수 있으므로 class imbalance 문제가 발생할 수 있다.

따라서 다음 지표를 함께 사용한다.

| 지표 | 의미 |
|---|---|
| Accuracy | 전체 분류 정확도 |
| Macro F1 | 클래스 불균형을 고려한 평균 F1 |
| Transition Precision | 모델이 transition이라고 예측한 것 중 실제 transition 비율 |
| Transition Recall | 실제 transition 중 모델이 탐지한 비율 |
| Transition F1 | 전이 구간 탐지 핵심 지표 |
| Params | 모델 파라미터 수 |
| Inference Time | window당 추론 시간 |
| Detection Latency | 실제 전이 발생 후 탐지까지 걸린 지연 |

---

## 9. 추가 실험

### 9.1 Sensor Ablation

가속도계와 자이로스코프의 기여도를 비교한다.

| 실험 | 입력 채널 |
|---|---|
| Acc only | acc_x, acc_y, acc_z |
| Gyro only | gyro_x, gyro_y, gyro_z |
| Acc + Gyro | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z |

#### 분석 목적

- 동작 전이 탐지에서 각속도 정보가 실제로 중요한지 확인한다.
- 회전성 신호가 전이 구간 탐지 성능에 미치는 영향을 분석한다.
- Mamba-3 기반 상태 추적 모델이 gyro 정보를 효과적으로 활용하는지 확인한다.

---

### 9.2 Window Length Ablation

window 길이에 따른 성능 변화를 분석한다.

| 조건 | 길이 | timestep |
|---|---:|---:|
| Short window | 1.28초 | 64 |
| Default window | 2.56초 | 128 |
| Long window | 5.12초 | 256 |

#### 분석 목적

- 전이 탐지에 필요한 시간 문맥의 길이를 확인한다.
- window가 너무 짧으면 전이 전후 맥락이 부족할 수 있다.
- window가 너무 길면 전이 시점이 희석될 수 있다.
- Mamba-3 기반 모델이 긴 sequence에서 성능과 효율성을 유지하는지 분석한다.

---

### 9.3 Subject-Independent Test

훈련 피험자와 테스트 피험자를 분리하여 일반화 성능을 평가한다.

```text
train subjects ≠ test subjects
```

#### 분석 목적

- 특정 사람의 움직임 패턴을 외운 것이 아니라 새로운 사용자에게도 일반화되는지 확인한다.
- 고령자 케어, 웨어러블 기기, 로봇 응용에서는 새로운 사용자에 대한 일반화가 중요하다.

---

## 10. 결과 표 구성 예시

2페이지 소논문에서는 표를 1개 중심으로 배치하는 것이 좋다.

```markdown
| Model | Acc | Macro F1 | Transition Precision | Transition Recall | Transition F1 | Params | Inference Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-CNN | - | - | - | - | - | - | - |
| GRU | - | - | - | - | - | - | - |
| TCN | - | - | - | - | - | - | - |
| Transformer Encoder | - | - | - | - | - | - | - |
| Mamba-3-based SSM | - | - | - | - | - | - | - |
```

### 10.1 현재 구현 기준 파라미터 수

현재 1차 실험 코드(`experiments/imu_transition/`) 기준 모델 크기는 다음과 같다.  
이 값들은 pretrained language model 크기(예: 130M, 1.3B, 2.8B)가 아니라, IMU binary transition detection용 소형 분류기 기준이다.

#### 입력 채널 3개 (`acc only` 또는 `gyro only`)

| Model | Params | Approx. |
|---|---:|---:|
| 1D-CNN | 75,906 | 0.0759M |
| GRU | 38,466 | 0.0385M |
| TCN | 220,610 | 0.2206M |
| Transformer Encoder | 83,842 | 0.0838M |
| Mamba-3-based SSM | 239,378 | 0.2394M |

#### 입력 채널 6개 (`acc + gyro`)

| Model | Params | Approx. |
|---|---:|---:|
| 1D-CNN | 76,866 | 0.0769M |
| GRU | 39,042 | 0.0390M |
| TCN | 221,378 | 0.2214M |
| Transformer Encoder | 84,034 | 0.0840M |
| Mamba-3-based SSM | 239,762 | 0.2398M |

#### 메모

- `acc only`와 `gyro only`는 모두 입력 채널 수가 3개이므로 파라미터 수가 동일하다.
- 현재 구현 기준에서는 대략 `GRU < CNN < Transformer < TCN < Mamba-3` 순으로 모델 크기가 증가한다.
- 따라서 본 실험의 baseline과 제안 모델은 모두 sub-million 규모의 소형 네트워크이며, B-scale language model과 직접 비교하는 설정은 아니다.

---

## 11. 기대되는 결과 해석 방향

실험 결과는 다음과 같은 방향으로 해석할 수 있다.

### Case 1. Mamba-3가 Transition F1에서 가장 좋은 경우

이 경우 논문 주장은 가장 강해진다.

```text
선택적 상태공간 모델이 IMU 시계열의 상태 변화 구간을 효과적으로 추적할 수 있음을 확인하였다.
```

### Case 2. Mamba-3가 Accuracy는 낮지만 Transition Recall/F1이 높은 경우

이 경우에도 논문 주장이 가능하다.

```text
전체 activity classification에서는 기존 모델과 유사하거나 낮은 성능을 보였으나,
전이 구간 탐지에서는 더 높은 recall 또는 F1을 보여 상태 변화 탐지에 적합한 가능성을 보였다.
```

### Case 3. Transformer가 성능은 가장 좋고 Mamba-3가 더 빠른 경우

이 경우 성능-효율 trade-off를 주장할 수 있다.

```text
Transformer는 높은 탐지 성능을 보였으나 추론 비용이 컸고,
Mamba-3 기반 모델은 유사한 전이 탐지 성능을 더 낮은 추론 시간으로 달성하였다.
```

### Case 4. Mamba-3가 baseline보다 좋지 않은 경우

이 경우에도 2페이지 데모 논문에서는 다음과 같이 정리할 수 있다.

```text
IMU 기반 동작 전이 탐지에서 선택적 상태공간 모델의 가능성을 검토하였으나,
현재 설정에서는 TCN 또는 Transformer 기반 모델이 더 안정적인 성능을 보였다.
향후 Mamba-3 구조의 sensor-specific adaptation 및 window-level transition labeling 개선이 필요하다.
```

---

## 12. 2페이지 소논문 구성안

2페이지 데모용 논문은 다음 구조를 추천한다.

```text
1. 초록
2. 서론
3. 관련 연구
4. 실험 설계
5. 예상 결과 또는 예비 결과
6. 결론
```

### 12.1 초록

- IMU 기반 동작 인식의 활용성
- 기존 HAR의 한계
- 동작 전이 구간 탐지의 필요성
- Mamba-3 기반 선택적 상태공간 모델 적용
- 비교 모델 및 평가 지표 요약

### 12.2 서론

- walking, sitting, standing만 인식해서는 부족하다는 문제 제기
- 로봇, 재활, 고령자 케어, 낙상 감지에서 전이 시점 감지가 중요하다는 응용 배경
- 전이 구간은 짧고 불안정하며 라벨 혼재 문제가 있다는 난점
- 효율적인 상태 추적 시계열 모델이 필요하다는 논리 전개

### 12.3 관련 연구

- IMU 기반 HAR
- CNN/RNN/TCN/Transformer 기반 시계열 인식
- 선택적 상태공간 모델 및 Mamba 계열 모델

### 12.4 실험 설계

- 데이터셋
- 라벨 재정의
- 입력 구성
- 비교 모델
- 평가 지표
- sensor ablation

### 12.5 결과 및 분석

- 모델별 결과표
- Transition F1 중심 해석
- 추론 시간 및 파라미터 비교
- sensor ablation 결과 해석

### 12.6 결론

- IMU 기반 동작 인식을 정적 분류에서 전이 탐지 문제로 확장
- 선택적 상태공간 모델의 가능성 검토
- 향후 subject-independent evaluation, 긴 sequence 평가, 실제 로봇 응용으로 확장 가능

---

## 13. 실험 실행 우선순위

실험을 빠르게 진행하려면 아래 순서로 진행한다.

### 1단계. 데이터셋 로딩 및 라벨 재정의

- UCI Postural Transition 데이터셋 다운로드
- raw inertial signal 로딩
- 기본 동작 라벨 → non-transition
- 자세 전이 라벨 → transition

### 2단계. Baseline 구현

- 1D-CNN
- GRU
- TCN
- Transformer Encoder

### 3단계. Mamba-3 기반 모델 구현

- 가능하면 Mamba-3 기반 selective SSM block 사용
- 구현이 어려우면 Mamba/Mamba-2 기반 block으로 먼저 대체
- 논문에서는 “Mamba-3 기반 구조로 확장 예정” 또는 “선택적 상태공간 모델 기반”으로 표현 가능

### 4단계. 핵심 지표 산출

- Accuracy
- Macro F1
- Transition Precision
- Transition Recall
- Transition F1
- Params
- Inference Time

### 5단계. Ablation 실험

- acc only
- gyro only
- acc + gyro

### 6단계. 2페이지 논문 작성

- 표 1개
- 방법 그림 1개
- 서론/관련 연구/실험 설계 중심으로 작성

---

## 14. 최소 실험 세트

시간이 부족할 경우 아래만 수행해도 2페이지 소논문 형태는 만들 수 있다.

```text
Dataset:
- UCI Smartphone-Based HAR with Postural Transitions

Task:
- Binary transition detection

Input:
- 128 timestep × 6 channels
- acc + gyro

Models:
- 1D-CNN
- GRU
- TCN
- Transformer Encoder
- Mamba-3-based SSM

Metrics:
- Accuracy
- Macro F1
- Transition Precision
- Transition Recall
- Transition F1
- Inference Time

Ablation:
- acc only
- gyro only
- acc + gyro
```

---

## 15. 참고 링크

- UCI Smartphone-Based Recognition of Human Activities and Postural Transitions  
  https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions

- UCI Human Activity Recognition Using Smartphones  
  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

- PAMAP2 Physical Activity Monitoring  
  https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

- OPPORTUNITY Activity Recognition  
  https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces  
  https://arxiv.org/abs/2312.00752

- Mamba-3: Enhanced State Space Models  
  https://arxiv.org/abs/2603.15569
