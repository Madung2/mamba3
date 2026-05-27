# Mamba-3 Encoder와 GyroPhase Head를 활용한 IMU 동작 전이 방향 추정 연구

> 수정 초안 v2  
> 핵심 방향: **TIC 논문의 Transformer encoder 역할을 Mamba-3/Complex SSM encoder로 재해석**하고, hidden phase·gyro magnitude·rotation diversity를 활용하는 **GyroPhase Head**를 제안한다.

---

## 초록

IMU(Inertial Measurement Unit) 기반 동작 인식은 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 연구는 주로 일정 길이의 시간 window를 입력으로 받아 walking, sitting, standing과 같은 현재 동작 클래스를 분류하는 Human Activity Recognition(HAR) 문제에 초점을 두어 왔다. 그러나 실제 응용 환경에서는 현재 동작의 종류뿐 아니라, 사용자의 상태가 언제 어떤 방향으로 변화하는지를 빠르고 안정적으로 감지하는 것이 중요하다. 특히 보조 로봇이나 재활 로봇은 사용자가 이미 앉아 있는지 또는 서 있는지를 분류하는 것만으로는 충분하지 않으며, 일어나려는지, 앉으려는지, 균형을 잃고 있는지와 같은 상태 전이의 방향과 시점을 조기에 파악해야 한다.

본 연구는 IMU 동작 전이 탐지를 회전성·위상성 시계열 표현 문제로 재정의하고, Mamba-3/Complex SSM을 단순 classifier가 아니라 **짧은 IMU history에서 회전성 상태 변화를 인코딩하는 temporal state encoder**로 사용한다. 이 관점은 Transformer IMU Calibrator(TIC)가 짧은 IMU history를 Transformer encoder로 인코딩하여 동적 calibration parameter를 추정하고, Rotation Diversity가 충분한 window에서만 보정값을 갱신한 접근에서 출발한다. 본 연구는 이를 동작 전이 방향 추정 문제로 확장하여, hidden phase 변화량, gyro magnitude, rotation diversity를 명시적으로 사용하는 **GyroPhase Head**를 제안한다.

예비 실험에서 UCI HAPT 데이터셋의 표준 transition 및 direction classification 성능은 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났으며, vanilla Mamba-3와 naive Complex-SSM은 이를 능가하지 못하였다. 그러나 학습된 Complex-SSM의 hidden state phase 변화량은 입력 gyro magnitude와 강한 양의 상관을 보였고, 전이 구간에서 비전이 구간보다 일관되게 크게 변화하였다. 이는 complex-valued state가 회전성 IMU 신호를 표현 수준에서 추적할 수 있음을 시사한다. 본 연구는 이러한 표현 수준의 회전성 정보를 전이 방향 분류 성능으로 연결하기 위해 GyroPhase Head를 설계하고, Transformer encoder baseline, Mamba-3 encoder baseline, TCN baseline과 비교하여 그 효과를 검증한다.

---

## 1. 서론

IMU 센서는 가속도계와 자이로스코프를 통해 사용자의 움직임을 연속적으로 측정할 수 있으며, 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 작업자 안전 관리, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 IMU 기반 동작 인식 연구는 주로 일정 길이의 시간 window를 입력으로 받아 walking, sitting, standing, lying과 같은 현재 동작 클래스를 분류하는 HAR 문제로 다루어져 왔다. 이러한 접근은 정적인 동작 구간을 구분하는 데 효과적이지만, 실제 환경에서 필요한 정보는 단순히 “현재 어떤 동작인가”에 그치지 않는다.

특히 로봇 및 인간-로봇 상호작용 환경에서는 사용자의 현재 상태뿐 아니라, 상태가 변화하는 순간을 빠르게 인식하는 것이 중요하다. 예를 들어 보조 로봇이나 재활 로봇은 사용자가 서 있는 상태인지 앉아 있는 상태인지만 아는 것으로는 충분하지 않다. 사용자가 일어나려는지, 앉으려는지, 균형을 잃고 있는지, 보행 중 회전하거나 멈추려는지를 조기에 파악해야 안전한 보조 동작을 수행할 수 있다. 고령자 케어와 낙상 감지에서도 낙상 이후의 lying 상태를 분류하는 것보다, 정상 보행에서 균형 상실 또는 급격한 자세 변화로 이어지는 전이 구간을 빠르게 감지하는 것이 더 중요하다. 즉, 실제 응용에서는 동작 클래스 자체보다 동작 간 전이 시점과 변화 양상을 인식하는 능력이 핵심이 된다.

그러나 동작 전이 구간은 일반적인 정적 동작 구간보다 인식이 어렵다. 전이 구간은 지속 시간이 짧고, 하나의 window 안에 두 개 이상의 동작 특성이 혼재할 수 있으며, 개인별 움직임 속도와 센서 부착 위치, 센서 노이즈의 영향을 크게 받는다. 예를 들어 walking에서 sitting으로 전환되는 과정은 보행 감속, 몸통 회전, 무릎 굽힘, 착석 충격 등 여러 신호가 연속적으로 나타나며, 이 과정은 사람마다 다른 시간 길이와 패턴을 가진다. 따라서 고정 길이 window 기반의 단순 분류 모델은 전이 시점을 늦게 감지하거나, 일시적인 흔들림을 실제 동작 변화로 오인할 수 있다.

기존 시계열 모델인 CNN, RNN, TCN, Transformer는 IMU 기반 동작 인식에 널리 활용되어 왔다. CNN과 TCN은 지역적인 센서 패턴을 효과적으로 추출할 수 있으며, 특히 dilated convolution 기반 TCN은 짧은 IMU window에서 강한 practical baseline이 될 수 있다. 반면 Transformer는 self-attention을 통해 window 내 전역 문맥을 활용할 수 있지만, 긴 시계열에서 계산량이 증가하고 사용자 독립 일반화에서 불안정할 수 있다. 상태공간 모델(State Space Model, SSM)과 Mamba 계열 모델은 입력에 따라 hidden state를 갱신하고 장기 시퀀스를 효율적으로 처리할 수 있다는 장점이 있으나, 단순히 Mamba-3를 classifier backbone으로 사용하는 것만으로는 짧은 IMU window에서 TCN/CNN을 능가하지 못할 수 있다.

본 연구는 이 지점에서 Mamba-3의 역할을 재정의한다. Mamba-3를 “최종 분류 성능이 가장 좋은 모델”로 주장하는 대신, 짧은 IMU history에서 회전성 상태 변화를 인코딩하는 **temporal state encoder**로 사용한다. 이는 Transformer IMU Calibrator(TIC)가 짧은 IMU orientation과 acceleration history를 Transformer encoder로 읽어 coordinate drift와 sensor-to-bone offset을 추정한 구조와 연결된다. TIC에서 중요한 점은 Transformer 자체보다, 최근 IMU window의 회전 정보가 현재 상태 추정에 중요하며, Rotation Diversity가 충분한 구간에서만 업데이트를 적용한다는 설계이다.

본 연구는 이 관점을 동작 전이 탐지로 확장한다. 사용자의 자세 전이 구간 역시 짧은 window 내에서 각속도 변화와 회전 다양성을 포함한다. 따라서 이를 효과적으로 인코딩할 수 있는 temporal state encoder가 필요하며, 복소수 상태 업데이트는 hidden state의 크기 변화와 위상 변화를 함께 표현할 수 있으므로 gyro 기반 회전성 신호를 hidden phase 변화로 추적할 가능성이 있다. 그러나 기존 average pooling 또는 linear classifier head는 이러한 phase 정보를 명시적으로 사용하지 않는다. 이에 본 연구는 hidden magnitude, hidden phase 변화량, gyro magnitude, rotation diversity를 결합하는 **GyroPhase Head**를 제안하여, 표현 수준에서 관찰된 회전성 정보를 전이 방향 분류 성능으로 연결하고자 한다.

본 연구의 기여는 다음과 같다.

1. IMU 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장한다.
2. TIC의 short-history encoder 및 Rotation Diversity 관점을 동작 전이 탐지 문제에 도입한다.
3. Mamba-3/Complex SSM을 회전성 상태 변화를 인코딩하는 temporal state encoder로 재정의한다.
4. hidden phase 변화량, gyro magnitude, rotation diversity를 명시적으로 사용하는 GyroPhase Head를 제안한다.
5. Transformer encoder baseline, Mamba-3 encoder baseline, TCN baseline, GyroPhase Head ablation을 통해 제안 구조의 효과와 한계를 검증한다.

---

## 2. 관련 연구

### 2.1 IMU 기반 Human Activity Recognition

IMU 기반 HAR은 가속도계와 자이로스코프에서 수집한 다변량 시계열 신호를 이용하여 사용자의 동작 상태를 분류하는 문제이다. 기존 연구들은 주로 일정 길이의 sliding window를 구성한 뒤, 각 window에 대해 walking, standing, sitting, lying과 같은 동작 라벨을 예측하는 방식으로 접근하였다. 이러한 window 기반 분류 방식은 정적인 동작 구간에서는 효과적이지만, 동작이 변화하는 경계 구간에서는 하나의 window 안에 서로 다른 동작 특성이 섞일 수 있어 분류가 불안정해질 수 있다. 또한 실제 응용에서는 단순한 동작 클래스 분류보다 동작이 언제 변화하는지, 어떤 방향으로 변화하는지, 변화가 얼마나 빠르게 발생하는지를 인식하는 것이 중요하다.

### 2.2 IMU short-history encoding과 동적 calibration

최근 Transformer IMU Calibrator(TIC)는 sparse inertial motion capture에서 기존 static calibration의 한계를 지적하고, 짧은 IMU history를 입력으로 받아 동적으로 변화하는 calibration parameter를 추정하는 접근을 제안하였다. TIC의 핵심은 IMU coordinate drift와 sensor-to-bone offset이 전체 sequence 동안 고정된다고 가정하지 않고, 짧은 window 안에서는 거의 일정하다고 보는 것이다. 또한 해당 window 내 IMU reading이 충분히 다양한 회전을 포함해야 calibration parameter 추정이 신뢰할 수 있다고 보고, Rotation Diversity trigger를 사용하여 충분한 회전 관측이 있는 경우에만 보정값을 갱신한다.

이 연구는 IMU 시계열에서 짧은 history와 회전 다양성이 현재 상태 추정에 중요하다는 점을 보여준다. 그러나 TIC의 목적은 calibration parameter regression이며, 동작 전이 탐지에서 hidden representation이 gyro 기반 회전성 신호를 어떻게 표현하고 classifier가 이를 어떻게 활용하는지는 다루지 않는다. 본 연구는 이 관점을 동작 전이 탐지 문제로 확장한다.

### 2.3 CNN, TCN, Transformer 기반 시계열 모델

IMU 시계열 인식을 위해 CNN, TCN, RNN, Transformer 등 다양한 딥러닝 기반 시계열 모델이 사용되어 왔다. CNN 기반 모델은 짧은 시간 구간의 지역 패턴을 효과적으로 추출할 수 있으며, TCN은 dilated convolution을 활용하여 더 넓은 시간 문맥을 반영할 수 있다. RNN과 LSTM, GRU는 순차적인 상태 변화를 모델링할 수 있다는 장점이 있으나, 긴 시퀀스 처리와 학습 효율성 측면에서 한계가 있다. Transformer는 self-attention을 통해 전역 문맥을 활용할 수 있지만, 긴 센서 시계열에서는 연산량과 메모리 사용량이 증가한다. 기존 실험에서도 짧은 128 timestep IMU window에서는 TCN과 1D-CNN이 매우 강한 baseline으로 나타났다.

### 2.4 선택적 상태공간 모델과 복소수 상태 업데이트

상태공간 모델은 연속적인 시계열 데이터를 hidden state의 변화로 표현하는 구조를 가지며, 긴 시퀀스를 효율적으로 처리할 수 있다는 장점이 있다. 최근 선택적 상태공간 모델은 입력에 따라 상태 갱신 방식을 조절함으로써 중요한 정보를 선택적으로 유지하거나 갱신하는 방향으로 발전하고 있다. Mamba 계열 모델은 이러한 선택적 상태공간 모델의 대표적인 예이며, Mamba-3는 complex-valued state update를 통해 hidden state의 크기와 위상 변화를 함께 표현할 수 있는 구조적 가능성을 제공한다.

복소수 곱셈은 크기 조절과 위상 회전을 동시에 표현할 수 있으므로, 회전성·주기성·방향성 변화가 포함된 시계열에 대해 구조적 편향을 제공할 가능성이 있다. IMU 동작 전이 문제에서는 자이로스코프 신호가 각속도와 자세 전환 정보를 포함하므로, 복소수 상태공간 표현이 이러한 회전성 신호를 hidden phase 변화로 추적할 수 있는지 검토할 필요가 있다.

### 2.5 본 연구의 위치

기존 IMU 동작 인식 연구는 대체로 가속도와 자이로스코프 신호를 동일한 입력 채널로 결합하여 분류 모델에 전달한다. 이 방식은 높은 분류 성능을 달성할 수 있지만, 자이로스코프가 포함하는 회전성 정보가 모델 내부에서 어떤 방식으로 표현되는지, 그리고 classifier가 이를 명시적으로 활용하는지는 충분히 분석되지 않았다.

본 연구는 다음과 같은 위치를 가진다. 첫째, IMU 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장한다. 둘째, TIC의 short-history encoder 및 Rotation Diversity 관점을 참고하여, 동작 전이 window 내 회전성 정보가 충분한지 판단하는 feature를 도입한다. 셋째, Mamba-3/Complex SSM을 단순 classifier가 아니라 회전성 상태 변화를 인코딩하는 temporal state encoder로 사용한다. 넷째, hidden phase 변화량, gyro magnitude, rotation diversity를 classifier에 명시적으로 전달하는 GyroPhase Head를 제안한다.

---

## 3. 문제 정의

### 3.1 입력 시계열

길이 `T`의 IMU window를 다음과 같이 정의한다.

```text
X = {x_1, x_2, ..., x_T}
```

각 timestep의 입력은 다음 6개 채널로 구성된다.

```text
x_t = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]_t
```

여기서 acceleration은 선형 움직임 및 충격 변화를 반영하고, gyroscope는 각속도와 자세 전환에 따른 회전성 변화를 반영한다.

### 3.2 Task A: Binary transition detection

각 window가 동작 전이 구간을 포함하는지 예측한다.

```text
y_bin ∈ {non-transition, transition}
```

### 3.3 Task B: 7-class transition direction classification

각 window의 전이 방향을 다음 7개 class로 예측한다.

```text
0: non-transition
1: stand-to-sit
2: sit-to-stand
3: sit-to-lie
4: lie-to-sit
5: stand-to-lie
6: lie-to-stand
```

본 연구의 핵심 task는 7-class transition direction classification이다. 단순 전이 여부보다, 실제 로봇·재활 응용에서 사용자가 어떤 방향으로 상태를 바꾸는지 아는 것이 더 중요하기 때문이다.

### 3.4 Rotation-aware subset

본 연구는 전체 test set뿐 아니라 다음 subset에서도 성능을 분석한다.

```text
High-gyro subset
Low-gyro subset
High-RD subset
Low-RD subset
High-gyro & High-RD subset
```

이는 GyroPhase Head가 회전성 정보가 강한 구간에서 더 효과적인지 확인하기 위함이다.

---

## 4. 제안 방법

## 4.1 TIC-style temporal encoder baseline

먼저 Transformer encoder를 IMU short-history encoder baseline으로 둔다.

```text
IMU sequence
    ↓
Transformer Encoder
    ↓
Temporal Average Pooling
    ↓
Linear Classifier
```

이 구조는 TIC에서 Transformer가 IMU short history를 인코딩하여 calibration parameter를 추정한 역할을, 본 연구의 전이 방향 분류 문제에 맞게 단순화한 baseline이다.

---

## 4.2 Mamba-3 temporal state encoder

Mamba-3 encoder는 Transformer encoder를 대체하는 temporal state encoder로 사용한다.

```text
IMU sequence
    ↓
Mamba-3 Encoder
    ↓
Temporal Average Pooling
    ↓
Linear Classifier
```

이 모델은 Mamba-3가 Transformer와 동일한 IMU short-history encoding 역할을 수행할 수 있는지 확인하기 위한 baseline이다.

---

## 4.3 Complex hidden state와 phase

Mamba-3 또는 Complex SSM의 hidden state가 복소수 형태라고 할 때, hidden state를 다음과 같이 둔다.

```text
z_t = Re(z_t) + i Im(z_t)
```

hidden magnitude는 다음과 같다.

```text
m_t = |z_t| = sqrt(Re(z_t)^2 + Im(z_t)^2)
```

hidden phase는 다음과 같이 정의한다.

```text
φ_t = atan2(Im(z_t), Re(z_t))
```

연속 시점 간 phase 변화량은 다음과 같다.

```text
Δφ_t = wrap(φ_t - φ_{t-1})
```

여기서 `wrap(·)`은 phase 차이를 `[-π, π]` 범위로 정규화하는 함수이다.

---

## 4.4 Gyro magnitude

자이로스코프 magnitude는 다음과 같이 정의한다.

```text
g_t = sqrt(gx_t^2 + gy_t^2 + gz_t^2)
```

이는 window 내 회전성 신호의 강도를 나타낸다.

---

## 4.5 Rotation Diversity

TIC의 Rotation Diversity 아이디어를 전이 탐지 문제에 맞게 변형한다. Rotation Diversity는 window 안에서 IMU 회전 방향이 얼마나 다양하게 변했는지를 나타내는 값이다.

초기 구현에서는 다음의 간단한 gyro 기반 diversity를 사용한다.

```text
RD_std = std_t(gyro_x) + std_t(gyro_y) + std_t(gyro_z)
```

또는 gyro vector 방향을 unit vector로 정규화한 뒤, 방향 공간을 coarse bin으로 양자화하여 방문한 bin 수를 count하는 방식으로 확장한다.

```text
RD_bin = number of visited gyro-direction bins in a window
```

본 연구에서는 먼저 `RD_std`를 사용하고, 추가 실험에서 `RD_bin`을 비교한다.

---

## 4.6 GyroPhase Head

GyroPhase Head는 hidden phase 변화량, gyro magnitude, rotation diversity를 classifier에 명시적으로 전달한다.

각 timestep에서 다음 feature를 구성한다.

```text
m_t = |z_t|
p_t = |Δφ_t|
q_t = g_t
d = RD(window)
```

상호작용 항은 다음과 같다.

```text
r_t = p_t · q_t
s_t = p_t · d
u_t = q_t · d
v_t = p_t · q_t · d
```

최종 phase-aware feature는 다음과 같다.

```text
f_t = [m_t, p_t, q_t, r_t, s_t, u_t, v_t]
```

시간축 pooling을 통해 window-level phase representation을 만든다.

```text
h_phase = Pool_t(f_t)
```

`Pool_t`는 다음을 포함한다.

```text
mean pooling
max pooling
standard deviation pooling
```

기존 encoder의 window-level representation을 `h_base`라고 하면, 최종 classifier 입력은 다음과 같다.

```text
h_final = concat(h_base, h_phase)
```

최종 예측은 다음과 같이 계산한다.

```text
ŷ = Classifier(h_final)
```

---

## 4.7 학습 목적 함수

Binary transition detection과 direction classification을 함께 학습하는 경우 다음 loss를 사용한다.

```text
L = L_transition + λ L_direction
```

초기 실험에서는 `λ = 1.0`으로 둔다.

확장 실험에서는 boundary offset estimation을 추가하여 다음과 같이 확장할 수 있다.

```text
L = L_transition + λ1 L_direction + λ2 L_boundary
```

---

## 5. 실험 설계

### 5.1 데이터셋

본 연구는 UCI Smartphone HAR + Postural Transitions(HAPT) 데이터셋을 사용한다. 해당 데이터셋은 기본 동작과 자세 전이 동작을 포함하며, 3축 가속도계와 3축 자이로스코프 신호를 제공한다.

기본 설정은 다음과 같다.

```text
Sampling rate = 50Hz
Window length = 128 timestep
Stride = 64 timestep
Input channels = acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

---

### 5.2 비교 모델

| 그룹 | 모델 | 목적 |
|---|---|---|
| Baseline | 1D-CNN | lightweight local pattern baseline |
| Baseline | TCN | strongest practical temporal convolution baseline |
| Baseline | Transformer + AvgPool | TIC-style temporal encoder baseline |
| Baseline | Mamba-3 + AvgPool | Transformer 대체 가능성 확인 |
| Head ablation | Mamba-3 + Magnitude Head | hidden magnitude 효과 확인 |
| Head ablation | Mamba-3 + Phase Head | hidden phase 변화량 효과 확인 |
| Head ablation | Mamba-3 + GyroPhase Head | phase + gyro magnitude 효과 확인 |
| Head ablation | Mamba-3 + GyroPhase + RD Head | phase + gyro + rotation diversity 효과 확인 |
| Encoder ablation | Transformer + GyroPhase Head | head 효과가 Transformer에서도 나타나는지 확인 |
| Encoder ablation | Complex-SSM + GyroPhase Head | 복소수 hidden phase 직접 활용 |
| Encoder ablation | Real-SSM + comparable head | real-valued state baseline |

---

### 5.3 평가 지표

| Metric | 설명 |
|---|---|
| Accuracy | 전체 정확도 |
| Macro F1 | 전체 class 평균 F1 |
| Direction Macro F1 | transition direction class 1~6 평균 F1 |
| Transition F1 | binary로 변환했을 때 transition F1 |
| Worst-class F1 | 가장 낮은 transition class F1 |
| Opposite-pair confusion | 반대 방향 전이 혼동률 |
| Miss rate | 전이 segment를 한 번도 탐지하지 못한 비율 |
| Detection latency | 전이 시작 후 첫 탐지까지의 지연 |
| Inference ms/window | window당 추론 시간 |
| Params | 모델 파라미터 수 |
| Generalization drop | random split 대비 subject-independent 성능 하락 |

---

## 6. 실험 계획

### Experiment 1. TIC-style Transformer vs Mamba-3 Encoder

목적은 Transformer가 맡은 IMU short-history encoder 역할을 Mamba-3가 대체할 수 있는지 확인하는 것이다.

비교 모델:

```text
Transformer Encoder + AvgPool
Mamba-3 Encoder + AvgPool
```

성공 기준:

```text
Mamba-3 Encoder + AvgPool >= Transformer Encoder + AvgPool
```

---

### Experiment 2. GyroPhase Head 효과 검증

목적은 hidden phase 정보를 classifier에 명시적으로 넣으면 성능이 향상되는지 확인하는 것이다.

비교 모델:

```text
Mamba-3 + AvgPool
Mamba-3 + Magnitude Head
Mamba-3 + Phase Head
Mamba-3 + GyroPhase Head
Mamba-3 + GyroPhase + RD Head
```

성공 기준:

```text
Mamba-3 + GyroPhase Head > Mamba-3 + AvgPool
Mamba-3 + GyroPhase + RD Head >= Mamba-3 + GyroPhase Head
```

---

### Experiment 3. Transformer + GyroPhase Head 비교

목적은 GyroPhase Head의 효과가 Mamba-3에만 있는지, Transformer에도 적용 가능한 일반 head인지 확인하는 것이다.

비교 모델:

```text
Transformer + AvgPool
Transformer + GyroPhase Head
Mamba-3 + AvgPool
Mamba-3 + GyroPhase Head
```

해석:

- `Mamba-3 + GyroPhase > Transformer + GyroPhase`이면 Mamba encoder의 필요성이 강해진다.
- `Transformer + GyroPhase`도 개선되면 GyroPhase Head 자체가 backbone-independent한 phase-aware readout으로 의미를 가진다.
- 둘 다 개선되지 않으면 phase feature 설계 또는 Rotation Diversity 계산을 재검토해야 한다.

---

### Experiment 4. Rotation Diversity ablation

목적은 Rotation Diversity가 phase-aware classifier의 신뢰도를 높이는지 검증하는 것이다.

비교 모델:

```text
GyroPhase Head without RD
GyroPhase Head with RD
```

subset 분석:

```text
High-gyro subset
Low-gyro subset
High-RD subset
Low-RD subset
High-gyro & High-RD subset
```

기대 결과:

```text
High-gyro & High-RD subset에서 GyroPhase + RD Head의 성능이 가장 좋아야 한다.
Low-RD subset에서는 phase signal이 노이즈일 수 있으므로 RD가 false positive를 줄일 수 있다.
```

---

### Experiment 5. Opposite-pair confusion 분석

목적은 전이 방향 분류에서 반대 방향 전이를 덜 헷갈리는지 확인하는 것이다.

주요 pair:

```text
stand-to-sit vs sit-to-stand
sit-to-lie vs lie-to-sit
stand-to-lie vs lie-to-stand
```

성공 기준:

```text
GyroPhase Head가 AvgPool Head보다 opposite-pair confusion을 줄임
```

---

### Experiment 6. Subject-independent evaluation

목적은 사용자 독립 조건에서 제안 구조가 일반화되는지 확인하는 것이다.

비교 모델:

```text
TCN
Transformer + AvgPool
Mamba-3 + AvgPool
Mamba-3 + GyroPhase Head
Mamba-3 + GyroPhase + RD Head
```

핵심 지표:

```text
Direction Macro F1
Transition F1
Generalization drop
Miss rate
```

---

### Experiment 7. Synthetic rotation task

목적은 통제된 회전성 시계열에서 Mamba-3/GyroPhase 구조가 유리한지 확인하는 것이다.

Task:

```text
1. angular velocity change detection
2. phase jump detection
3. direction classification without explicit omega input
```

기존 synthetic direction task는 `ω`가 입력에 포함되어 모든 모델이 100%로 풀어버렸으므로, 새 direction task에서는 `ω`를 입력에서 제거하고 `[cos θ_t, sin θ_t]`만 제공한다.

---

## 7. 예비 결과 요약

### 7.1 표준 모델 비교

기존 5-seed 실험에서 128 timestep, acc+gyro 기준 표준 transition 및 direction classification에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났다. Mamba-3는 binary transition detection에서는 Transformer와 유사한 수준에 근접했으나, 7-class direction classification에서는 TCN을 능가하지 못하였다. 이는 짧은 IMU window에서 Mamba-3를 단순 classifier backbone으로 쓰는 것만으로는 충분하지 않음을 보여준다.

### 7.2 Hidden phase 분석

학습된 Complex-SSM의 hidden phase 변화량은 gyro magnitude와 강한 양의 상관을 보였고, 전이 구간에서 비전이 구간보다 더 크게 변화하였다. 이는 복소수 상태공간 모델이 회전성 신호를 내부 표현 수준에서 추적할 수 있음을 시사한다. 그러나 naive classifier head는 이 정보를 직접 활용하지 못해 direction classification 성능이 낮았다. 따라서 GyroPhase Head를 통해 hidden phase 정보를 classifier에 명시적으로 전달할 필요가 있다.

### 7.3 TIC 논문에서 가져온 구조적 시사점

TIC는 짧은 IMU history를 Transformer encoder로 인코딩하여 calibration parameter를 추정하고, Rotation Diversity가 충분한 경우에만 update를 적용한다. 본 연구는 이 아이디어를 전이 탐지에 맞게 변형한다. 즉, 회전성 정보가 충분히 포함된 window에서 hidden phase가 더 의미 있는 feature가 될 수 있으므로, gyro magnitude와 rotation diversity를 GyroPhase Head에 포함한다.

---

## 8. 결과 해석 시나리오

### Case A. Mamba-3 + GyroPhase가 Transformer보다 명확히 좋음

주장:

```text
Mamba-3는 IMU short-history encoder로 Transformer를 대체할 수 있으며,
GyroPhase Head는 hidden phase와 gyro/RD 정보를 활용해 전이 방향 분류를 개선한다.
```

논문각이 가장 강하다.

---

### Case B. Mamba-3 + GyroPhase는 Transformer보다 좋지만 TCN보다 낮음

주장:

```text
TCN은 여전히 가장 강한 practical baseline이지만,
Mamba-3 + GyroPhase는 Transformer 기반 temporal encoder보다 우수하며,
회전성 상태 표현을 활용하는 SSM 기반 encoder의 가능성을 보였다.
```

석사논문으로 가능하다.

---

### Case C. GyroPhase Head가 Mamba/Transformer 모두에서 성능 향상

주장:

```text
GyroPhase Head는 특정 backbone에 종속되지 않는 phase-aware readout으로,
IMU 동작 전이에서 gyro 기반 회전성 정보를 classifier에 전달하는 일반 구조로 볼 수 있다.
```

Mamba 중심성은 약해지지만, 제안 head 논문으로 가능하다.

---

### Case D. Mamba도 GyroPhase도 약하고 TCN만 강함

방향 전환:

```text
GyroPhase-TCN
```

주장:

```text
Phase-1/2 분석 결과, 짧은 IMU window에서는 TCN이 가장 안정적인 backbone이었다.
따라서 본 연구는 TCN backbone에 gyro/RD feature를 결합한 GyroPhase-TCN으로 방향을 전환한다.
```

---

## 9. 한계 및 향후 과제

본 연구의 hidden phase 분석은 complex-valued hidden state의 phase가 입력 gyro magnitude 및 전이 구간과 함께 변화하는 경향을 보고하는 해석적 분석이다. 이는 Mamba-3 또는 Complex-SSM이 실제 신체 회전을 직접 이해한다는 의미가 아니며, hidden phase가 실제 관절 각도나 IMU orientation과 1:1 대응된다는 의미도 아니다.

또한 기존 예비 실험에서 naive Complex-SSM은 direction classification에서 낮은 성능을 보였다. 따라서 복소수 상태 업데이트가 항상 분류 성능 향상으로 이어진다고 주장할 수 없다. 본 연구의 핵심은 hidden phase 정보가 내부 표현으로 형성될 수 있다는 관찰을 바탕으로, 이를 classifier가 명시적으로 활용하도록 만드는 구조를 제안하고 검증하는 것이다.

향후 연구에서는 다음을 수행한다.

1. GyroPhase Head를 적용한 direction classification 성능 검증
2. Rotation Diversity feature의 효과 분석
3. opposite-pair confusion 감소 여부 확인
4. subject-independent evaluation 확장
5. streaming setting에서 detection latency 재평가
6. Mamba-3 결과가 약할 경우 GyroPhase-TCN으로 전환

---

## 10. 결론

본 연구는 IMU 기반 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장한다. 기존 예비 실험에서 표준 분류 성능은 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났고, vanilla Mamba-3와 naive Complex-SSM은 이를 능가하지 못하였다. 그러나 Complex-SSM의 hidden phase 변화량은 gyro magnitude와 강한 상관을 보였고, 전이 구간에서 비전이 구간보다 더 크게 변화하였다. 이는 복소수 상태공간 표현이 회전성 IMU 신호를 내부적으로 추적할 수 있음을 시사한다.

본 연구는 TIC 논문의 short-history encoder와 Rotation Diversity trigger 관점을 동작 전이 탐지에 도입하여, Mamba-3를 IMU 회전성 상태 변화를 인코딩하는 temporal state encoder로 사용한다. 또한 hidden phase 변화량, gyro magnitude, rotation diversity를 명시적으로 사용하는 GyroPhase Head를 제안하여, 표현 수준의 회전성 정보를 전이 방향 분류 성능으로 연결하고자 한다. 이를 통해 본 연구는 Mamba-3를 단순 최신 backbone으로 사용하는 것이 아니라, IMU 회전성 상태 표현과 classifier head 사이의 구조적 연결을 설계하는 방향으로 확장한다.

---

## 참고문헌 메모

- Transformer IMU Calibrator: Dynamic On-body IMU Calibration for Inertial Motion Capture
- UCI Smartphone HAR + Postural Transitions
- Mamba / Mamba-3 selective state space model
- TCN: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
