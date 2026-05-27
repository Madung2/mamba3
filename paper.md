# IMU 동작 전이 탐지를 위한 복소수 상태공간 표현 분석 및 GyroPhase Head 연구

## 초록

IMU(Inertial Measurement Unit) 기반 동작 인식은 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 연구는 주로 일정 길이의 시간 window를 입력으로 받아 walking, sitting, standing과 같은 현재 동작 클래스를 분류하는 Human Activity Recognition(HAR) 문제에 초점을 두어 왔다. 그러나 실제 응용 환경에서는 현재 동작의 종류뿐 아니라, 사용자의 상태가 언제 어떤 방향으로 변화하는지를 빠르고 안정적으로 감지하는 것이 중요하다. 특히 보조 로봇이나 재활 로봇은 사용자가 이미 앉아 있는지 또는 서 있는지를 분류하는 것만으로는 충분하지 않으며, 일어나려는지, 앉으려는지, 균형을 잃고 있는지와 같은 상태 전이의 방향과 시점을 조기에 파악해야 한다.

본 연구는 IMU 동작 전이 탐지를 회전성·위상성 시계열 표현 문제로 재정의하고, 복소수 상태공간 모델(complex-valued state space model)이 자이로스코프 기반 회전성 신호를 내부 표현 수준에서 어떻게 추적하는지 분석한다. 표준 transition 및 direction classification 실험에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났으며, vanilla Mamba-3와 자체 구현한 naive Complex-SSM은 이를 능가하지 못하였다. 그러나 학습된 Complex-SSM의 hidden state phase 변화량은 입력 gyro magnitude와 강한 양의 상관을 보였고, 전이 구간에서 비전이 구간보다 일관되게 크게 변화하였다. 이는 complex-valued state가 회전성 IMU 신호를 표현 수준에서 추적할 수 있음을 시사한다.

기존 classifier head는 이러한 hidden phase 정보를 명시적으로 활용하지 않기 때문에, 표현 수준에서 관찰된 회전성 정보가 최종 전이 방향 분류 성능으로 충분히 연결되지 못할 수 있다. 이에 본 연구는 hidden magnitude, hidden phase 변화량, gyro magnitude를 명시적으로 결합하는 **GyroPhase Head**를 제안한다. 제안 구조는 복소수 상태공간 표현에서 추출한 phase-aware feature를 classifier에 직접 전달함으로써, 회전성 신호가 강한 동작 전이 구간과 전이 방향 분류에서 성능 향상을 목표로 한다. 본 연구는 IMU 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장하고, 복소수 상태공간 표현의 가능성과 한계를 분석하며, 이를 활용하는 위상 인식 분류 구조를 제안한다는 점에서 의의를 가진다.

---

## 1. 서론

IMU 센서는 가속도계와 자이로스코프를 통해 사용자의 움직임을 연속적으로 측정할 수 있으며, 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 작업자 안전 관리, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 IMU 기반 동작 인식 연구는 주로 일정 길이의 시간 window를 입력으로 받아 walking, sitting, standing, lying과 같은 현재 동작 클래스를 분류하는 HAR 문제로 다루어져 왔다. 이러한 접근은 정적인 동작 구간을 구분하는 데 효과적이지만, 실제 환경에서 필요한 정보는 단순히 “현재 어떤 동작인가”에 그치지 않는다.

특히 로봇 및 인간-로봇 상호작용 환경에서는 사용자의 현재 상태뿐 아니라, 상태가 변화하는 순간을 빠르게 인식하는 것이 중요하다. 예를 들어 보조 로봇이나 재활 로봇은 사용자가 서 있는 상태인지 앉아 있는 상태인지만 아는 것으로는 충분하지 않다. 사용자가 일어나려는지, 앉으려는지, 균형을 잃고 있는지, 보행 중 회전하거나 멈추려는지를 조기에 파악해야 안전한 보조 동작을 수행할 수 있다. 고령자 케어와 낙상 감지에서도 낙상 이후의 lying 상태를 분류하는 것보다, 정상 보행에서 균형 상실 또는 급격한 자세 변화로 이어지는 전이 구간을 빠르게 감지하는 것이 더 중요하다. 즉, 실제 응용에서는 동작 클래스 자체보다 동작 간 전이 시점과 변화 양상을 인식하는 능력이 핵심이 된다.

그러나 동작 전이 구간은 일반적인 정적 동작 구간보다 인식이 어렵다. 전이 구간은 지속 시간이 짧고, 하나의 window 안에 두 개 이상의 동작 특성이 혼재할 수 있으며, 개인별 움직임 속도와 센서 부착 위치, 센서 노이즈의 영향을 크게 받는다. 예를 들어 walking에서 sitting으로 전환되는 과정은 보행 감속, 몸통 회전, 무릎 굽힘, 착석 충격 등 여러 신호가 연속적으로 나타나며, 이 과정은 사람마다 다른 시간 길이와 패턴을 가진다. 따라서 고정 길이 window 기반의 단순 분류 모델은 전이 시점을 늦게 감지하거나, 일시적인 흔들림을 실제 동작 변화로 오인할 수 있다.

기존 시계열 모델인 CNN, RNN, TCN, Transformer는 IMU 기반 동작 인식에 널리 활용되어 왔다. CNN과 TCN은 지역적인 센서 패턴을 효과적으로 추출할 수 있으며, 특히 dilated convolution 기반 TCN은 짧은 IMU window에서 강한 practical baseline이 될 수 있다. 그러나 이러한 모델들은 보통 가속도와 자이로스코프 신호를 단순한 다채널 입력으로 결합하여 처리한다. 이 경우 자이로스코프가 포함하는 각속도, 자세 전환, 회전성 변화가 hidden representation에서 어떻게 표현되는지, 그리고 classifier가 이를 명시적으로 활용하는지는 충분히 드러나지 않는다.

본 연구는 이러한 한계에 주목하여 IMU 동작 전이를 회전성·위상성 시계열 표현 문제로 바라본다. 자이로스코프 신호는 사용자의 자세 전환과 회전성 변화를 직접적으로 반영한다. 복소수 상태 업데이트는 hidden state 공간에서 크기 변화와 위상 변화를 함께 표현할 수 있으므로, 각속도·자세 전환처럼 회전성 또는 주기성이 포함된 IMU 시계열을 표현하는 데 구조적 편향(inductive bias)을 제공할 가능성이 있다. 그러나 복소수 상태공간 모델의 hidden phase가 회전성 신호를 표현하더라도, 기존 classifier head가 이를 명시적으로 활용하지 못한다면 최종 전이 방향 분류 성능으로 이어지지 않을 수 있다.

이에 본 연구는 먼저 Mamba-3 기반 구조와 자체 구현한 Real-SSM/Complex-SSM ablation block, 그리고 synthetic 회전 task를 사용하여 복소수 상태공간 표현의 가능성과 한계를 분석한다. 이후 학습된 Complex-SSM의 hidden phase 변화량과 gyro magnitude의 관계를 정량적으로 분석하고, 이를 바탕으로 hidden phase 정보를 classifier에 직접 전달하는 **GyroPhase Head**를 제안한다. 본 연구의 목표는 Mamba-3 또는 Complex-SSM이 모든 baseline보다 우수함을 주장하는 것이 아니라, IMU 동작 전이 문제에서 회전성 신호가 내부 표현으로 어떻게 형성되고, 이를 어떻게 분류 성능으로 연결할 수 있는지를 실험적으로 검토하는 것이다.

---

## 2. 관련 연구

### 2.1 IMU 기반 Human Activity Recognition

IMU 기반 Human Activity Recognition은 가속도계와 자이로스코프에서 수집한 다변량 시계열 신호를 이용하여 사용자의 동작 상태를 분류하는 문제이다. 기존 연구들은 주로 일정 길이의 sliding window를 구성한 뒤, 각 window에 대해 walking, standing, sitting, lying과 같은 동작 라벨을 예측하는 방식으로 접근하였다. 이러한 window 기반 분류 방식은 정적인 동작 구간에서는 효과적이지만, 동작이 변화하는 경계 구간에서는 하나의 window 안에 서로 다른 동작 특성이 섞일 수 있어 분류가 불안정해질 수 있다. 또한 실제 응용에서는 단순한 동작 클래스 분류보다 동작이 언제 변화하는지, 어떤 방향으로 변화하는지, 변화가 얼마나 빠르게 발생하는지를 인식하는 것이 중요하다.

### 2.2 시계열 모델 기반 동작 인식

IMU 시계열 인식을 위해 CNN, RNN, TCN, Transformer 등 다양한 딥러닝 기반 시계열 모델이 사용되어 왔다. CNN 기반 모델은 짧은 시간 구간의 지역 패턴을 효과적으로 추출할 수 있으며, TCN은 dilated convolution을 활용하여 더 넓은 시간 문맥을 반영할 수 있다. RNN과 LSTM, GRU는 순차적인 상태 변화를 모델링할 수 있다는 장점이 있으나, 긴 시퀀스 처리와 학습 효율성 측면에서 한계가 있다. Transformer는 self-attention을 통해 전역 문맥을 활용할 수 있지만, 긴 센서 시계열에서는 연산량과 메모리 사용량이 증가한다. 이처럼 기존 모델들은 각각 장점을 가지지만, 동작 전이 구간처럼 짧은 변화와 긴 문맥이 동시에 중요한 문제에서는 성능, 일반화, 지연 특성을 함께 고려할 필요가 있다.

### 2.3 선택적 상태공간 모델과 복소수 상태 업데이트

상태공간 모델(State Space Model, SSM)은 연속적인 시계열 데이터를 상태의 변화로 표현하는 구조를 가지며, 긴 시퀀스를 효율적으로 처리할 수 있다는 장점이 있다. 최근 선택적 상태공간 모델은 입력에 따라 상태 갱신 방식을 조절함으로써 중요한 정보를 선택적으로 유지하거나 갱신하는 방향으로 발전하고 있다. Mamba 계열 모델은 이러한 선택적 상태공간 모델의 대표적인 예로, 긴 시퀀스에서 효율적인 처리를 목표로 한다.

특히 Mamba-3 및 관련 복소수 상태공간 모델은 complex-valued state update를 통해 hidden state의 크기와 위상 변화를 함께 표현할 수 있다. 복소수 곱셈은 크기 조절과 위상 회전을 동시에 표현할 수 있으므로, 회전성·주기성·방향성 변화가 포함된 시계열에 대해 구조적 편향을 제공할 가능성이 있다. IMU 동작 전이 문제에서는 자이로스코프 신호가 각속도와 자세 전환 정보를 포함하므로, 복소수 상태공간 표현이 이러한 회전성 신호를 내부 phase 변화로 추적할 수 있는지 검토할 필요가 있다.

### 2.4 기존 연구의 한계와 본 연구의 위치

기존 IMU 동작 인식 연구는 대체로 가속도와 자이로스코프 신호를 동일한 입력 채널로 결합하여 분류 모델에 전달한다. 이 방식은 높은 분류 성능을 달성할 수 있지만, 자이로스코프가 포함하는 회전성 정보가 모델 내부에서 어떤 방식으로 표현되는지, 그리고 classifier가 이를 명시적으로 활용하는지는 충분히 분석되지 않았다. 또한 복소수 상태공간 모델의 phase 표현이 실제 IMU 회전성 신호와 어떤 관계를 가지는지에 대한 정량적 분석도 부족하다.

본 연구는 다음과 같은 위치를 가진다. 첫째, IMU 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장한다. 둘째, 복소수 상태공간 모델의 hidden phase 변화량이 자이로스코프 기반 회전성 신호와 어떤 관계를 가지는지 분석한다. 셋째, 표현 수준에서 관찰된 hidden phase 정보를 최종 classifier가 명시적으로 활용하도록 하는 GyroPhase Head를 제안한다. 이를 통해 본 연구는 단순한 모델 성능 비교를 넘어, 회전성 신호의 내부 표현과 분류 구조 사이의 연결을 다룬다.

---

## 3. 본 연구의 기여 및 핵심 주장

본 연구는 UCI HAPT 데이터셋과 통제된 synthetic 회전 시계열을 함께 사용하여 총 304 runs를 수행하였으며, 다음을 정량적으로 보고한다.

1. **표준 분류 성능 분석**  
   acc+gyro 입력 기준으로 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났다. TCN은 binary transition detection에서 Transition F1 0.9649 ± 0.0148, 7-class direction classification에서 direction macro F1 0.8118 ± 0.0211을 보였으며, Mamba-3는 각각 0.9609 ± 0.0145, 0.7150 ± 0.0908을 기록하였다. 짧은 IMU window 설정에서는 선택적 상태공간 모델의 즉시적 우위가 관찰되지 않았다.

2. **사용자 독립 일반화 분석**  
   학습/검증/테스트 사용자를 완전히 분리한 subject-independent 평가에서 TCN은 random split 대비 Δ Transition F1 = 0%의 손실로 가장 견고하였으며, Transformer는 −3.3%p로 가장 큰 일반화 손실을 보였다. Mamba-3는 −1.5%p의 중간 수준 일반화 손실을 보였다.

3. **Synthetic 회전 시계열 분석**  
   통제된 synthetic 회전 시계열에서 angular velocity change 탐지 task를 수행한 결과, Mamba-3가 macro F1 0.9162 ± 0.0215로 TCN, Real-SSM, Transformer 등 모든 baseline을 능가하였다. 이는 선택적 상태공간 모델이 회전 동역학의 변화 추적이 필요한 조건에서 상대적 이점을 가질 수 있음을 시사한다.

4. **Hidden phase 분석**  
   학습된 Complex-SSM의 마지막 layer hidden state에서 phase를 추출하고 윈도우당 평균 `|Δphase|`를 계산한 결과, 전이 구간에서 비전이 구간 대비 평균 1.231 ± 0.057배 더 큰 phase 변화량을 보였으며, 입력 gyro magnitude와 Pearson 상관 r = 0.847 ± 0.020을 10/10 run에서 일관되게 보였다. 이는 complex-valued state가 회전성 신호를 표현 수준에서 추적하는 inductive bias를 형성한다는 정량적 근거이다.

5. **GyroPhase Head 제안**  
   기존 classifier head는 복소수 hidden state의 phase 정보를 명시적으로 사용하지 않는다. 본 연구는 hidden magnitude, hidden phase 변화량, gyro magnitude를 결합하는 GyroPhase Head를 제안하여, 표현 수준에서 관찰된 회전성 정보를 전이 탐지 및 전이 방향 분류 성능으로 연결하고자 한다.

### 3.1 핵심 주장

본 연구의 핵심 주장은 다음과 같다.

> 표준 transition 및 direction classification에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났으며, vanilla Mamba-3와 naive Complex-SSM은 이를 능가하지 못하였다. 그러나 학습된 Complex-SSM의 hidden phase 변화량은 입력 gyro magnitude와 강한 양의 상관을 보였고, 전이 구간에서 비전이 구간보다 일관되게 크게 변화하였다. 이는 complex-valued state가 회전성 IMU 신호를 표현 수준에서 추적할 수 있음을 시사한다. 본 연구는 이러한 관찰을 바탕으로 hidden phase 변화량과 gyro magnitude를 명시적으로 사용하는 GyroPhase Head를 제안하여, 표현 수준의 회전성 정보를 전이 방향 분류 성능으로 연결하고자 한다.

### 3.2 해석상의 주의

본 연구의 hidden phase 분석은 complex-valued hidden state의 phase가 입력 gyro magnitude 및 전이 구간과 함께 변화하는 경향을 보고하는 해석적 분석이며, 다음과 같이 과해석해서는 안 된다.

- Mamba-3 또는 Complex-SSM이 실제 신체 회전을 직접 이해한다는 의미가 아니다.
- Complex hidden state의 phase가 실제 관절 각도 또는 IMU 자세 quaternion에 1:1 대응된다는 의미도 아니다.
- 본 연구의 결과는 복소수 상태 업데이트가 회전성·주기성·방향성 변화가 포함된 IMU 시계열을 표현하는 데 구조적 편향을 제공할 수 있으며, 학습된 hidden state에서 그 편향이 부분적으로 관찰된다는 약한 형태의 가설을 지지한다.

---

## 4. 제안 방법: GyroPhase Head

### 4.1 복소수 상태 업데이트

복소수 상태공간 모델의 hidden state를 다음과 같이 정의한다.

```text
z_t = a_t ⊙ z_{t-1} + b_t ⊙ u_t
```

여기서 `z_t ∈ C^d`는 complex hidden state, `u_t`는 입력 projection, `a_t`는 상태 유지 및 회전 계수, `b_t`는 입력 주입 계수이다. 복소수 계수 `a_t`는 다음과 같이 크기와 위상으로 표현할 수 있다.

```text
a_t = ρ_t · exp(i θ_t)
```

따라서 상태 업데이트는 다음과 같이 쓸 수 있다.

```text
z_t = ρ_t · exp(i θ_t) ⊙ z_{t-1} + b_t ⊙ u_t
```

이때 `ρ_t`는 hidden state의 magnitude scaling을, `θ_t`는 hidden state의 phase rotation을 조절한다. 이 구조는 입력 시계열의 크기 변화와 위상 변화를 함께 표현할 수 있다.

### 4.2 Hidden phase와 gyro magnitude

복소수 hidden state `z_t`의 실수부와 허수부를 각각 `Re(z_t)`, `Im(z_t)`라고 할 때, hidden phase는 다음과 같이 정의한다.

```text
φ_t = atan2(Im(z_t), Re(z_t))
```

연속 시점 간 phase 변화량은 다음과 같이 계산한다.

```text
Δφ_t = wrap(φ_t - φ_{t-1})
```

여기서 `wrap(·)`은 phase 차이를 `[-π, π]` 범위로 정규화하는 함수이다. 자이로스코프 magnitude는 다음과 같이 정의한다.

```text
g_t = sqrt(gx_t^2 + gy_t^2 + gz_t^2)
```

본 연구의 예비 분석에서 윈도우별 평균 `|Δφ_t|`는 윈도우별 평균 `g_t`와 강한 양의 상관을 보였으며, 전이 구간에서 비전이 구간보다 더 크게 나타났다. 이는 hidden phase 변화량이 회전성 IMU 신호와 관련된 정보를 포함할 가능성을 보여준다.

### 4.3 GyroPhase feature

본 연구는 기존 classifier head가 hidden phase 정보를 명시적으로 사용하지 않는다는 점에 주목한다. 이를 해결하기 위해 다음과 같은 phase-aware feature를 구성한다.

```text
m_t = |z_t|
p_t = |Δφ_t|
q_t = g_t
r_t = p_t · q_t
```

여기서 `m_t`는 hidden magnitude, `p_t`는 hidden phase 변화량, `q_t`는 gyro magnitude, `r_t`는 phase 변화와 gyro magnitude의 상호작용 항이다.

이후 시간축 pooling을 통해 window-level phase-aware representation을 만든다.

```text
h_phase = Pool_t([m_t, p_t, q_t, r_t])
```

`Pool_t`는 mean pooling, max pooling, standard deviation pooling을 포함할 수 있다.

### 4.4 최종 classifier

기존 backbone에서 얻은 window-level representation을 `h_base`라고 할 때, GyroPhase Head는 다음과 같이 최종 representation을 구성한다.

```text
h_final = concat(h_base, h_phase)
```

최종 예측은 다음과 같이 계산한다.

```text
ŷ = Classifier(h_final)
```

본 연구에서는 `ŷ`를 다음 두 task에 대해 사용한다.

```text
1. binary transition detection
2. 7-class transition direction classification
```

확장 구조에서는 activity classification과 boundary offset estimation을 함께 수행하는 multi-task head로 확장할 수 있다.

### 4.5 학습 목적 함수

기본 학습 목적 함수는 다음과 같다.

```text
L = L_transition + λ L_direction
```

여기서 `L_transition`은 binary transition detection cross-entropy loss이고, `L_direction`은 transition direction classification cross-entropy loss이다. 초기 실험에서는 `λ = 1.0`으로 설정한다.

향후 boundary offset을 함께 예측하는 경우 다음과 같이 확장할 수 있다.

```text
L = L_transition + λ1 L_direction + λ2 L_boundary
```

---

## 5. 실험 설계

### 5.1 데이터셋

본 연구는 UCI Smartphone HAR + Postural Transitions(HAPT) 데이터셋을 사용한다. 해당 데이터셋은 30명의 피험자가 스마트폰을 착용한 상태에서 수행한 기본 동작과 자세 전이 동작을 포함하며, 3축 가속도계와 3축 자이로스코프 신호를 제공한다.

### 5.2 Task 정의

본 연구에서는 다음 task를 사용한다.

1. **Binary transition detection**
   - non-transition vs transition

2. **7-class transition direction classification**
   - non-transition
   - stand-to-sit
   - sit-to-stand
   - sit-to-lie
   - lie-to-sit
   - stand-to-lie
   - lie-to-stand

3. **Synthetic rotation task**
   - rotation direction
   - phase jump detection
   - angular velocity change detection

### 5.3 비교 모델

비교 모델은 다음과 같다.

| 모델 | 역할 |
|---|---|
| 1D-CNN | lightweight local pattern baseline |
| GRU | recurrent sequential baseline |
| TCN | strong practical temporal convolution baseline |
| Transformer Encoder | attention-based global context baseline |
| Mamba-3 | selective SSM baseline |
| Real-SSM | real-valued state update baseline |
| Complex-SSM | complex-valued state update ablation |
| Complex-SSM + GyroPhase Head | proposed phase-aware structure |

### 5.4 평가 지표

본 연구에서는 Accuracy만으로는 전이 탐지 성능을 충분히 평가할 수 없기 때문에 다음 지표를 함께 사용한다.

| 지표 | 설명 |
|---|---|
| Accuracy | 전체 정확도 |
| Macro F1 | 클래스 불균형을 고려한 평균 F1 |
| Transition F1 | transition class에 대한 F1 |
| Direction Macro F1 | transition direction class에 대한 macro F1 |
| Worst-class F1 | 가장 성능이 낮은 transition class의 F1 |
| Miss rate | 전이 segment 중 한 번도 탐지되지 않은 비율 |
| Detection latency | 전이 시작 이후 첫 탐지까지의 지연 |
| Inference time | window당 추론 시간 |
| Params | 모델 파라미터 수 |

---

## 6. 예비 결과 및 해석

### 6.1 Phase-1: 표준 모델 비교

5-seed 평균 실험 결과, 표준 128-window setting에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났다. Mamba-3는 binary transition detection에서 Transition F1 0.9609 ± 0.0145를 기록하여 TCN 및 Transformer를 명확히 능가하지 못하였다. Direction classification에서도 TCN이 direction macro F1 0.8118 ± 0.0211로 가장 높은 성능을 보였고, Mamba-3는 0.7150 ± 0.0908에 머물렀다.

이 결과는 짧은 IMU window 기반 전이 탐지에서는 복잡한 선택적 상태공간 모델보다 convolution 기반 baseline이 더 실용적일 수 있음을 보여준다. 따라서 본 연구는 Mamba-3 또는 Complex-SSM의 단순 적용이 아니라, 회전성 표현을 명시적으로 활용하는 구조가 필요하다고 본다.

### 6.2 Phase-2: Hidden phase 분석

학습된 Complex-SSM의 hidden state를 분석한 결과, hidden phase 변화량은 gyro magnitude와 강한 양의 상관을 보였다. 또한 전이 구간에서는 비전이 구간보다 phase 변화량이 더 크게 나타났다. 이는 complex-valued state가 분류 성능과 별개로 회전성 신호를 내부 표현 수준에서 추적할 수 있음을 시사한다.

그러나 naive Complex-SSM은 direction classification에서 낮은 성능을 보였다. 이는 hidden phase 정보가 존재하더라도 기존 classifier head가 이를 명시적으로 활용하지 못할 수 있음을 의미한다. 따라서 본 연구는 hidden phase 변화량과 gyro magnitude를 classifier에 직접 전달하는 GyroPhase Head를 제안한다.

### 6.3 제안 구조 검증 계획

제안 구조는 다음 비교를 통해 검증한다.

| 모델 | 목적 |
|---|---|
| Complex-SSM | 기존 complex hidden state baseline |
| Complex-SSM + magnitude pooling | magnitude 정보만 추가한 경우 |
| Complex-SSM + phase pooling | hidden phase 변화량 추가 |
| Complex-SSM + GyroPhase Head | phase 변화량과 gyro magnitude 상호작용 포함 |
| TCN | 강한 practical baseline |
| Real-SSM | real-valued SSM baseline |

핵심 검증 질문은 다음과 같다.

1. GyroPhase Head가 naive Complex-SSM의 direction classification 성능을 회복시키는가?
2. GyroPhase Head가 Real-SSM 또는 Mamba-3 수준 이상의 direction macro F1을 달성하는가?
3. 회전성 전이가 강한 class 또는 high-gyro subset에서 성능 개선이 더 크게 나타나는가?
4. opposite-pair confusion, 예를 들어 sit-to-stand와 stand-to-sit, lie-to-sit와 lie-to-stand의 혼동이 줄어드는가?

---

## 7. 한계 및 향후 과제

본 연구는 복소수 상태공간 모델의 hidden phase가 IMU 회전성 신호와 관련된 정보를 포함할 수 있음을 보였으나, 이를 곧바로 실제 신체 회전각의 해석으로 연결할 수는 없다. Hidden phase는 모델 내부 표현이며, 실제 관절 각도나 IMU orientation과 1:1 대응하지 않는다. 또한 naive Complex-SSM은 direction classification에서 낮은 성능을 보였기 때문에, 복소수 상태 업데이트가 항상 분류 성능 향상으로 이어진다고 주장할 수 없다.

향후 연구에서는 GyroPhase Head를 통해 hidden phase 정보를 classifier에 명시적으로 전달하고, 전이 방향 분류 및 high-gyro subset에서 성능 개선 여부를 검증할 필요가 있다. 또한 leave-one-subject-out cross-validation, streaming evaluation, boundary offset estimation 등을 통해 실제 로봇 및 웨어러블 응용에 가까운 평가를 수행할 예정이다.

---

## 8. 결론

본 연구는 IMU 기반 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장하였다. 표준 분류 성능에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났으며, vanilla Mamba-3와 naive Complex-SSM은 이를 능가하지 못하였다. 그러나 Complex-SSM의 hidden phase 변화량은 gyro magnitude와 강한 상관을 보였고, 전이 구간에서 비전이 구간보다 더 크게 변화하였다. 이는 complex-valued state가 회전성 IMU 신호를 표현 수준에서 추적할 수 있음을 시사한다.

본 연구는 이러한 관찰을 바탕으로 hidden phase 변화량과 gyro magnitude를 명시적으로 사용하는 GyroPhase Head를 제안하였다. 향후 실험에서는 제안 head가 전이 방향 분류 성능과 회전성 전이 구간의 탐지 성능을 개선할 수 있는지 검증할 예정이다. 이를 통해 본 연구는 복소수 상태공간 표현의 회전성 inductive bias를 단순한 해석 분석에 그치지 않고, 실제 IMU 동작 전이 분류 구조로 연결하는 방향을 제시한다.
