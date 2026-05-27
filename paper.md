## 초록

IMU(Inertial Measurement Unit) 기반 동작 인식은 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 연구는 주로 walking, sitting, standing과 같은 현재 동작 클래스를 분류하는 Human Activity Recognition 문제에 초점을 두어 왔으나, 실제 응용 환경에서는 사용자의 상태가 변화하는 전이 구간을 빠르고 안정적으로 감지하는 것이 더 중요하다. 본 연구는 이 문제를 회전성·위상성 시계열 표현 관점에서 재정의하고, **IMU 동작 전이 탐지에서 complex-valued state update가 회전성 시계열 표현에 어떤 가능성과 한계를 가지는지** 실험적으로 분석한다.

UCI HAPT 데이터셋에서 5-seed 평균으로 평가한 결과, 표준 transition 및 direction 분류 성능에서는 **dilated TCN과 1D-CNN이 가장 안정적인 baseline** 으로 나타났으며 (TCN direction macro F1 0.812 ± 0.021), Mamba-3 및 자체 구현한 Complex-SSM은 이를 능가하지 못하였다. 그러나 (i) 통제된 synthetic 회전 시계열의 **angular velocity change 탐지** 에서 Mamba-3는 가장 높은 macro F1 (0.916 ± 0.022)을 보였고, (ii) 학습된 Complex-SSM의 **hidden state phase 변화량은 입력 gyro magnitude와 r=0.85 ± 0.02의 강한 양의 상관**을, 전이 구간에서는 비전이 대비 **1.23배 더 큰 변화량**을 10/10 run에서 일관되게 보였다. 이는 complex-valued state가 회전성 신호를 표현 수준에서 추적하는 inductive bias를 실제로 형성함을 의미한다. classifier head가 이 phase signal을 분류에 활용하도록 하는 구조적 개선이 향후 과제이다.

본 연구는 IMU 동작 인식을 정적 행동 분류에서 동작 전이 탐지 문제로 확장하고, 복소수 상태공간 모델의 회전성 표현 가능성과 한계를 정량적으로 검토한다는 점에서 의의를 가진다.

## 1. 서론

IMU(Inertial Measurement Unit) 센서는 가속도계와 자이로스코프를 통해 사용자의 움직임을 연속적으로 측정할 수 있으며, 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 작업자 안전 관리, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 IMU 기반 동작 인식 연구는 주로 일정 길이의 시간 window를 입력으로 받아 walking, sitting, standing, lying과 같은 현재 동작 클래스를 분류하는 Human Activity Recognition(HAR) 문제로 다루어져 왔다. 이러한 접근은 정적인 동작 구간을 구분하는 데 효과적이지만, 실제 환경에서 필요한 정보는 단순히 “현재 어떤 동작인가”에 그치지 않는다.

특히 로봇 및 인간-로봇 상호작용 환경에서는 사용자의 현재 상태뿐 아니라, 상태가 변화하는 순간을 빠르게 인식하는 것이 중요하다. 예를 들어 보조 로봇이나 재활 로봇은 사용자가 서 있는 상태인지 앉아 있는 상태인지만 아는 것으로는 충분하지 않다. 사용자가 일어나려는지, 앉으려는지, 균형을 잃고 있는지, 보행 중 회전하거나 멈추려는지를 조기에 파악해야 안전한 보조 동작을 수행할 수 있다. 고령자 케어와 낙상 감지에서도 낙상 이후의 lying 상태를 분류하는 것보다, 정상 보행에서 균형 상실 또는 급격한 자세 변화로 이어지는 전이 구간을 빠르게 감지하는 것이 더 중요하다. 즉, 실제 응용에서는 동작 클래스 자체보다 동작 간 전이 시점과 변화 양상을 인식하는 능력이 핵심이 된다.

그러나 동작 전이 구간은 일반적인 정적 동작 구간보다 인식이 어렵다. 전이 구간은 지속 시간이 짧고, 하나의 window 안에 두 개 이상의 동작 특성이 혼재할 수 있으며, 개인별 움직임 속도와 센서 부착 위치, 센서 노이즈의 영향을 크게 받는다. 예를 들어 walking에서 sitting으로 전환되는 과정은 보행 감속, 몸통 회전, 무릎 굽힘, 착석 충격 등 여러 신호가 연속적으로 나타나며, 이 과정은 사람마다 다른 시간 길이와 패턴을 가진다. 따라서 고정 길이 window 기반의 단순 분류 모델은 전이 시점을 늦게 감지하거나, 일시적인 흔들림을 실제 동작 변화로 오인할 수 있다.

기존 시계열 모델인 CNN, RNN, TCN, Transformer는 IMU 기반 동작 인식에 널리 활용되어 왔다. CNN과 TCN은 지역적인 센서 패턴을 효과적으로 추출할 수 있지만, 긴 시간 동안 유지되는 상태와 짧은 전이 변화를 함께 다루는 데 한계가 있을 수 있다. RNN 계열 모델은 순차 상태를 다룰 수 있으나 긴 시퀀스 학습과 병렬 처리 효율성 측면에서 제약이 있다. Transformer는 전역 문맥을 활용할 수 있지만, 시퀀스 길이가 길어질수록 연산 비용이 증가한다. 따라서 동작 전이 탐지와 같이 짧은 변화와 긴 문맥을 동시에 고려해야 하는 문제에서는, 효율적인 시계열 처리와 안정적인 상태 추적 능력을 함께 갖춘 모델이 필요하다.

본 연구는 IMU 시계열에서 동작 전이 구간을 탐지하기 위해 선택적 상태공간 모델, 특히 **complex-valued state update**의 가능성을 분석한다. 복소수 상태 업데이트는 hidden state 공간에서 크기 변화(magnitude)와 위상 변화(phase)를 함께 표현할 수 있으므로, 각속도·자세 전환처럼 회전성 또는 주기성이 포함된 IMU 시계열을 표현하는 데 구조적 편향(inductive bias)을 제공할 가능성이 있다. 단, 이는 "Mamba-3가 모든 모델보다 좋다"는 주장이 아니라 "complex-valued state update가 회전성 시계열 표현에 어떤 가능성과 한계를 가지는지" 검토하는 방향성이다. 본 연구에서는 Mamba-3 기반 구조와 자체 구현한 Real-SSM/Complex-SSM ablation block, 그리고 통제된 synthetic 회전 task를 함께 사용하여 기존 CNN, RNN, TCN, Transformer 계열 모델과 비교하고, 학습된 hidden state의 phase 분석을 통해 회전성 inductive bias가 표현 수준에서 어떻게 드러나는지 정량적으로 확인한다.

## 2. 관련 연구

### 2.1 IMU 기반 Human Activity Recognition

IMU 기반 Human Activity Recognition(HAR)은 가속도계와 자이로스코프에서 수집한 다변량 시계열 신호를 이용하여 사용자의 동작 상태를 분류하는 문제이다. 기존 연구들은 주로 일정 길이의 sliding window를 구성한 뒤, 각 window에 대해 walking, standing, sitting, lying과 같은 동작 라벨을 예측하는 방식으로 접근하였다. 이러한 window 기반 분류 방식은 정적인 동작 구간에서는 효과적이지만, 동작이 변화하는 경계 구간에서는 하나의 window 안에 서로 다른 동작 특성이 섞일 수 있어 분류가 불안정해질 수 있다. 또한 실제 응용에서는 단순한 동작 클래스 분류보다 동작이 언제 변화하는지, 변화가 얼마나 빠르게 발생하는지를 인식하는 것이 중요하다.

### 2.2 시계열 모델 기반 동작 인식

IMU 시계열 인식을 위해 CNN, RNN, TCN, Transformer 등 다양한 딥러닝 기반 시계열 모델이 사용되어 왔다. CNN 기반 모델은 짧은 시간 구간의 지역 패턴을 효과적으로 추출할 수 있으며, TCN은 dilated convolution을 활용하여 더 넓은 시간 문맥을 반영할 수 있다. RNN과 LSTM, GRU는 순차적인 상태 변화를 모델링할 수 있다는 장점이 있으나, 긴 시퀀스 처리와 학습 효율성 측면에서 한계가 있다. Transformer는 self-attention을 통해 전역 문맥을 활용할 수 있지만, 긴 센서 시계열에서는 연산량과 메모리 사용량이 증가한다. 이처럼 기존 모델들은 각각 장점을 가지지만, 동작 전이 구간처럼 짧은 변화와 긴 문맥이 동시에 중요한 문제에서는 성능과 효율성을 함께 고려할 필요가 있다.

### 2.3 선택적 상태공간 모델

상태공간 모델(State Space Model, SSM)은 연속적인 시계열 데이터를 상태의 변화로 표현하는 구조를 가지며, 긴 시퀀스를 효율적으로 처리할 수 있다는 장점이 있다. 최근 선택적 상태공간 모델은 입력에 따라 상태 갱신 방식을 조절함으로써 중요한 정보를 선택적으로 유지하거나 갱신하는 방향으로 발전하고 있다. 이러한 특성은 IMU 기반 동작 전이 탐지와 잘 연결된다. 동작 전이 구간에서는 기존 상태가 유지되다가 특정 시점에서 급격히 변화하므로, 모델은 불필요한 노이즈를 억제하면서 실제 상태 변화가 발생하는 순간을 포착해야 한다. 본 연구는 이러한 관점에서 Mamba-3 기반 선택적 상태공간 모델을 IMU 동작 전이 탐지 문제에 적용하고, 기존 시계열 모델과의 비교를 통해 전이 구간 탐지 가능성을 분석한다.


## 3. 본 연구의 기여 및 핵심 주장

본 연구는 UCI HAPT 데이터셋과 통제된 synthetic 회전 시계열을 함께 사용하여 총 304 runs (5-seed × 모델 × 채널 × 조건)을 수행하였으며, 다음을 정량적으로 보고한다.

1. **표준 분류 성능**: dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났다. acc+gyro 입력 기준으로 TCN은 binary transition detection에서 Transition F1 0.9649 ± 0.0148, 7-class direction classification에서 direction macro F1 0.8118 ± 0.0211을 보였으며, Mamba-3 (각각 0.9609 ± 0.0145, 0.7150 ± 0.0908)와 Transformer (0.9731 ± 0.0069, 0.6999 ± 0.0213)는 이를 능가하지 못하였다. 본 짧은 IMU window (128 timestep) 설정에서는 선택적 상태공간 모델의 즉시적 우위가 관찰되지 않는다.

2. **Subject-independent 일반화**: 학습/검증/테스트 사용자를 완전히 분리한 leave-subjects-out 평가에서 TCN은 random split 대비 Δ Transition F1 = 0% 의 손실로 가장 견고하였으며, Transformer는 −3.3%p로 가장 큰 일반화 손실을 보였다. Mamba-3는 중간 수준 (−1.5%p) 의 일반화 손실을 보였다.

3. **Synthetic 회전 시계열 (통제 실험)**: 입력에 ω가 포함된 direction task는 모든 모델이 100% 정확도로 해결하여 변별력이 없었으나, **angular velocity change 탐지**에서 Mamba-3가 macro F1 0.9162 ± 0.0215로 모든 baseline (TCN 0.9017, Real-SSM 0.8793, Transformer 0.8802, CNN 0.5896, Complex-SSM 0.5684)을 명확히 능가하였다. Phase jump detection은 TCN/CNN/Mamba-3/Real-SSM 모두 ≥0.987의 macro F1을 보여 변별력이 없었다.

4. **Hidden phase 분석 (본 연구의 핵심 관찰)**: 학습된 Complex-SSM의 마지막 layer hidden state에서 `phase = atan2(imag, real)` 를 추출하고 윈도우당 평균 `|Δphase|` 를 계산한 결과, (i) 전이 구간에서 비전이 구간 대비 평균 **1.231 ± 0.057배 더 큰 phase 변화량**, (ii) 입력 gyro magnitude (√(gx²+gy²+gz²))와의 **Pearson 상관 r = 0.847 ± 0.020** 을 10/10 run에서 일관되게 보였다. classifier의 direction 분류 성능과 무관하게, complex-valued state는 회전성 신호를 표현 수준에서 추적하는 inductive bias를 형성한다는 정량적 증거이다.

### 3.1 핵심 주장 (main claim)

> 표준 transition / direction 분류 성능에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났으며 (TCN direction macro F1 0.812 ± 0.021), Mamba-3 및 자체 Complex-SSM 구현은 이를 능가하지 못하였다. 그러나 (i) 통제된 synthetic 회전 시계열의 angular velocity change 탐지에서 Mamba-3는 가장 높은 macro F1 (0.916 ± 0.022)을 보였고, (ii) 본 연구에서 학습된 Complex-SSM의 hidden state phase 변화량은 입력 gyro magnitude와 r=0.85 ± 0.02의 강한 양의 상관을, 전이 구간에서 비전이 대비 1.23배 더 큰 변화량을 일관되게 보여, complex-valued state가 회전성 신호를 표현 수준에서 추적하는 inductive bias를 실제로 형성함을 확인하였다. classifier head가 이 phase signal을 분류에 활용하도록 하는 구조적 개선이 향후 과제이다.

### 3.2 해석상의 주의

본 연구의 hidden phase 분석은 complex-valued hidden state의 phase가 입력 gyro magnitude 및 전이 구간과 함께 변화하는 *경향*을 보고하는 해석적 분석이며, 다음과 같이 과해석해서는 안 된다.

- "Mamba-3 또는 Complex-SSM이 실제 신체 회전을 직접 이해한다" 는 의미가 아니다.
- "Complex hidden state의 phase가 실제 관절 각도 또는 IMU 자세 quaternion에 1:1 대응된다"는 의미도 아니다.
- 본 연구의 결과는 "복소수 상태 업데이트가 회전성·주기성·방향성 변화가 포함된 IMU 시계열을 표현하는 데 구조적 편향을 제공할 수 있으며, 학습된 hidden state에서 그 편향이 부분적으로 관찰된다"는 약한 형태의 가설을 지지한다.
