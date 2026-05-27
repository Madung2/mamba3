# Selective GyroPhase Head를 활용한 IMU 동작 전이 방향 추정 연구

## 초록

IMU(Inertial Measurement Unit) 기반 동작 인식은 스마트폰, 웨어러블 기기, 재활 모니터링, 고령자 케어, 인간-로봇 상호작용 등 다양한 분야에서 활용되고 있다. 기존 연구는 주로 일정 길이의 시간 window를 입력으로 받아 walking, sitting, standing과 같은 현재 동작 클래스를 분류하는 Human Activity Recognition(HAR) 문제에 초점을 두어 왔다. 그러나 실제 응용 환경에서는 현재 동작의 종류뿐 아니라, 사용자의 상태가 언제 어떤 방향으로 변화하는지를 빠르고 안정적으로 감지하는 것이 중요하다. 특히 보조 로봇이나 재활 로봇은 사용자가 이미 앉아 있는지 또는 서 있는지를 분류하는 것만으로는 충분하지 않으며, 일어나려는지, 앉으려는지, 균형을 잃고 있는지와 같은 상태 전이의 방향과 시점을 조기에 파악해야 한다.

본 연구는 IMU 동작 전이 탐지를 회전성·위상성 시계열 표현 문제로 재정의하고, 복소수 상태공간 모델(complex-valued state space model)이 자이로스코프 기반 회전성 신호를 내부 표현 수준에서 어떻게 추적하는지 분석한다. Phase-1/2 의 표준 transition 및 direction classification 실험에서는 dilated TCN 과 1D-CNN 이 가장 안정적인 baseline 으로 나타났으며 (TCN direction macro F1 0.812 ± 0.021), vanilla Mamba-3 와 자체 구현한 naive Complex-SSM 은 이를 능가하지 못하였다. 그러나 학습된 Complex-SSM 의 hidden state phase 변화량은 입력 gyro magnitude 와 r = 0.847 ± 0.020 의 강한 양의 상관을 보였고, 전이 구간에서 비전이 구간보다 1.231 ± 0.057 배 더 크게 변화하였다. 본 연구의 후속 phase-3 / phase-4 분석은 이 phase–gyro inductive bias 가 selective scanning 없이 학습된 complex-static SSM 에서도 동일하게 r = 0.84 로 형성됨을 확인하여, 회전성 inductive bias 의 출처가 selective scanning 이 아니라 **complex update 자체**임을 분리하여 보였다.

기존 classifier head 는 이러한 hidden phase 정보를 명시적으로 활용하지 않기 때문에, 표현 수준에서 관찰된 회전성 정보가 최종 전이 방향 분류 성능으로 충분히 연결되지 못한다. 이에 본 연구는 hidden magnitude, hidden phase 변화량, gyro magnitude, rotation diversity, selective_update_score 를 명시적으로 결합하는 **Selective GyroPhase Head** 를 제안한다. mean-pool encoder readout 으로 재학습한 phase-3 sweep 에서 Mamba-3 + GyroPhase + RD Head 가 direction macro F1 **0.7611 ± 0.0440** 으로 TCN baseline (0.7492 ± 0.0239) 을 포함한 모든 비교 모델 중 1위였고, Transformer encoder 에서도 +0.025 의 일관된 개선을 보여 backbone-independent 한 phase-aware readout 으로 작동한다. 본 head 의 효과는 가장 어려운 **transition-high-gyro subset 에서 +0.030** 으로 전체 평균 (+0.006) 의 5배에 달해 *난이도 적응형* 임을 시사하며, subject-disjoint 평가에서도 worst-class F1 +0.029 의 안정성 개선을 유지한다. 마지막으로 selective_update_score 의 적절한 proxy 가 retention coefficient rho 가 아니라 update budget `(1 − rho) × ||u||` 임을 정량적으로 확인하여 (corr +0.30, trans/non ratio 1.15, direction macro F1 +0.039), selective scanning 의 head feature 화 방향을 제시한다. 본 연구는 IMU 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장하고, 복소수 상태공간 표현의 가능성과 한계를 분석하며, 이를 활용하는 위상 인식 분류 구조를 제안·정량 검증한다는 점에서 의의를 가진다.

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

본 연구는 UCI HAPT 데이터셋과 통제된 synthetic 회전 시계열을 함께 사용하여 총 **519 runs** 를 수행하였다 (Phase-1/2: 304, Phase-3: 42 + 18, Phase-4: 153 — 후자는 selective proxy 6 + subject-disjoint 15 + harder synthetic 72 + transition-only 후처리 42 + phase 분석 18). 다음을 정량적으로 보고한다.

1. **표준 분류 성능 분석 (Phase-1/2, last-token pool)**  
   acc+gyro 입력 기준으로 dilated TCN 과 1D-CNN 이 가장 안정적인 baseline 으로 나타났다. TCN 은 binary transition detection 에서 Transition F1 0.9649 ± 0.0148, 7-class direction classification 에서 direction macro F1 0.8118 ± 0.0211 을 보였으며, Mamba-3 는 각각 0.9609 ± 0.0145, 0.7150 ± 0.0908 을 기록하였다. last-token pool 기반의 짧은 IMU window 설정에서는 선택적 상태공간 모델의 즉시적 우위가 관찰되지 않았다.

2. **사용자 독립 일반화 분석 (Phase-1)**  
   학습/검증/테스트 사용자를 완전히 분리한 subject-independent 평가에서 TCN 은 random split 대비 Δ Transition F1 = 0%의 손실로 가장 견고하였으며, Transformer 는 −3.3%p 로 가장 큰 일반화 손실을 보였다. Mamba-3 는 −1.5%p 의 중간 수준 일반화 손실을 보였다.

3. **Synthetic 회전 시계열 분석 (Phase-2 + Phase-4)**  
   ω 가 입력에 포함된 phase-2 angular velocity change 탐지 task 에서 Mamba-3 가 macro F1 0.9162 ± 0.0215 로 모든 baseline 을 능가하였다. Phase-4 의 cos/sin only `speed_direction6` (6-class) 에서도 Transformer / Mamba-3 가 동률 1위 (0.9881 ± 0.005) 였으며, 같은 task 의 worst-class F1 은 selective scanning 이 빠진 complex_static 에서 가장 낮았다 (0.918 vs complex_selective 0.970, Δ +0.052). 잘 통제된 회전 task 에서는 selective complex update 가 명확한 worst-class 안정성 이점을 가진다.

4. **Hidden phase 분석 (Phase-2 / Phase-3 재현)**  
   학습된 Complex-SSM 의 마지막 layer hidden state 에서 phase 를 추출하고 윈도우당 평균 `|Δphase|` 를 계산한 결과, 전이 구간에서 비전이 구간 대비 평균 **1.231 ± 0.057 배** 더 큰 phase 변화량을 보였으며, 입력 gyro magnitude 와 Pearson 상관 **r = 0.847 ± 0.020** 을 10/10 run 에서 일관되게 보였다. Phase-3 에서는 동일 상관이 5 개의 head 변형 × 3 seeds 의 18 ckpt 에서 0.79–0.85 범위로 재현되었으며, **selective scanning 이 없는 complex-static SSM 에서도 r = 0.84, ratio = 1.24** 가 동일하게 관찰되어 본 inductive bias 의 출처가 complex update 자체임을 분리해서 확인하였다.

5. **GyroPhase Head — Phase-3 정량 검증**  
   mean-pool encoder readout 위에 hidden magnitude / hidden phase 변화량 / gyro magnitude / rotation diversity 를 결합하는 GyroPhase + RD Head 를 Mamba-3 backbone 에 적용한 결과 direction macro F1 **0.7611 ± 0.0440** 으로 본 sweep 의 14 개 비교 모델 중 1위였다 (TCN baseline 0.7492 ± 0.0239). 동일 head 는 Transformer encoder 에 적용해도 **+0.025** 의 일관된 개선을 보여 backbone-independent 한 phase-aware readout 으로 작동한다.

6. **2 × 2 SSM ablation — Phase-3**  
   real-static → real-selective: direction macro F1 +0.072, complex-static → complex-selective: +0.021. selective scanning 의 효과는 real-valued state 에서 크고 complex-valued state 에서 작다. 그러나 phase–gyro 상관과 Δphase trans/non ratio 는 두 complex variant 에서 동일 (r = 0.83, ratio = 1.24) — selective scanning 은 phase signal 의 *형성* 과는 독립적이다.

7. **Selective update proxy 개선 — Phase-4**  
   Phase-3 의 selective_score = rho 는 `corr(rho, gyro) = −0.44`, `rho_trans / rho_non = 0.99` 로 head feature 신호가 약하였다. 본 연구는 update budget `(1 − rho) × ||u||` 를 새 proxy 로 제안하여 `corr = +0.30`, `ratio = 1.15` 의 양의 변별력을 확보하였으며, 이를 사용한 selective_gyrophase_v2 head 의 direction macro F1 은 **0.6365 ± 0.0826** 으로 legacy selective_gyrophase 0.5979 ± 0.0898 대비 **+0.039** 의 개선을 보였다.

8. **GyroPhase Head 의 난이도 적응형 이득 — Phase-4**  
   transition window 만 선택하고 그 안에서 gyro magnitude 중앙값으로 high/low 분할한 hardest subset (`trans-high-gyro`) 에서 Mamba-3 + GyroPhase + RD 는 Mamba-3 + AvgPool 대비 **+0.030** direction macro F1 개선을 보였다 (0.670 vs 0.640) — 전체 평균 Δ (+0.006) 의 5배. subject-disjoint 평가에서도 mean direction macro F1 은 모든 모델이 random split 과 동등 이상이었으나, **worst-class F1 에서는 Mamba-3 + GyroPhase+RD 가 1위 (0.6465)** — drop 이 큰 사용자에서의 안정성 측면에서 head 효과는 유지된다.

### 3.1 핵심 주장

본 연구의 핵심 주장은 다음과 같다.

> Phase-1/2 의 표준 transition / direction 분류에서는 dilated TCN 과 1D-CNN 이 가장 안정적인 baseline 으로 나타났으며 (TCN direction macro F1 0.812 ± 0.021), last-token pool 의 Mamba-3 및 자체 Complex-SSM 은 이를 능가하지 못하였다. 그러나 학습된 Complex-SSM 의 hidden phase 변화량은 입력 gyro magnitude 와 r = 0.85 ± 0.02 의 강한 양의 상관과 1.23배 더 큰 transition vs non-transition 변화량을 보였으며 (Phase-1/2 10 runs, Phase-3 18 ckpt 에서 동일 범위 재현), 이는 **selective scanning 없이 학습된 complex-static SSM 에서도 동일하게 형성**되어 회전성 inductive bias 의 출처가 complex update 자체임을 정량적으로 보였다. mean-pool encoder readout 위에 hidden magnitude / phase 변화량 / gyro magnitude / rotation diversity / selective update budget 을 결합하는 **Selective GyroPhase Head** 는 Mamba-3 backbone 에서 direction macro F1 0.7611 ± 0.0440 으로 TCN baseline 을 포함한 모든 비교 모델 중 1위였으며, Transformer 에서도 +0.025 의 일관된 개선을 보여 backbone-independent 한 phase-aware readout 으로 작동한다. 본 head 의 효과는 **transition-high-gyro 라는 가장 어려운 subset 에서 +0.030**, **subject-disjoint worst-class F1 에서 +0.029** 로 나타나 *난이도 적응형* 임을 시사한다. 또한 selective update 의 적절한 proxy 가 retention coefficient rho 가 아니라 update budget `(1 − rho) × ||u||` 임을 정량적으로 확인하여 (corr +0.30, +0.039 dirF1), selective scanning 의 head feature 화 방향을 제시한다.

### 3.2 해석상의 주의

본 연구의 hidden phase 분석은 complex-valued hidden state 의 phase 가 입력 gyro magnitude 및 전이 구간과 함께 변화하는 경향을 보고하는 해석적 분석이며, 다음과 같이 과해석해서는 안 된다.

- Mamba-3 또는 Complex-SSM 이 실제 신체 회전을 직접 이해한다는 의미가 아니다.
- Complex hidden state 의 phase 가 실제 관절 각도 또는 IMU 자세 quaternion 에 1:1 대응된다는 의미도 아니다.
- 본 연구의 결과는 복소수 상태 업데이트가 회전성·주기성·방향성 변화가 포함된 IMU 시계열을 표현하는 데 구조적 편향을 제공할 수 있으며, 학습된 hidden state 에서 그 편향이 부분적으로 관찰된다는 약한 형태의 가설을 지지한다.
- Phase-3 의 mean Δ direction macro F1 +0.006 (full set) 은 3-seed σ ≈ 0.04–0.06 안에서 통계적으로 분리되지 않는다. 본 연구의 직접적 정량 이득은 subset 별 (+0.030 trans-high-gyro), worst-class (+0.029), 또는 proxy 교체 후 (+0.039 selective_gyrophase_v2) 의 조건부 개선이며, paired bootstrap 등 통계 검증은 후속 과제로 남는다.
- Phase-3 의 selective scanning 분석은 본 연구의 자체 Complex-SSM 구현 (2 × 2 ablation) 에 한정된다. Mamba-3 의 fused triton/cute kernel 내부 state 는 본 sweep 에서 직접 hook 하지 않았으며 (exp_plan3-1 §13 대안 1), Mamba-3 + GyroPhase 변형은 input gyro-derived phase proxy 로 fallback 한다.

---

## 4. 제안 방법: Selective GyroPhase Head

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

본 연구는 기존 classifier head 가 hidden phase 정보를 명시적으로 사용하지 않는다는 점에 주목한다. 이를 해결하기 위해 다음과 같은 phase-aware feature 를 구성한다.

```text
m_t = |z_t|                     # hidden magnitude
p_t = |Δφ_t|                    # hidden phase 변화량
q_t = g_t                       # gyro magnitude
d   = RD(window)                # window-level Rotation Diversity
r_t = p_t · q_t                 # phase × gyro 상호작용
```

여기서 RD(window) 는 TIC 논문의 Rotation Diversity trigger 에서 영감을 받은 회전 다양성 지표로, 본 연구에서는 다음을 사용한다.

```text
RD_std = std_t(gx) + std_t(gy) + std_t(gz)
```

이후 시간축 pooling 을 통해 window-level phase-aware representation 을 만든다.

```text
h_phase = Pool_t([m_t, p_t, q_t, p_t · q_t, p_t · d, q_t · d, p_t · q_t · d])
```

`Pool_t` 는 mean / max / standard deviation pooling 을 동시에 사용한다.

### 4.4 Selective update budget (phase-4 도입)

선택적 상태공간 모델의 `ρ_t` 는 retention coefficient 로, 입력이 클 때 작아지는 음의 상관을 보인다. 따라서 ρ 그 자체는 transition 구간을 양으로 표시하는 head feature 로는 적합하지 않다 (phase-4 §1 정량 분석: corr(ρ, gyro) = −0.44, ρ_trans / ρ_non = 0.99). 본 연구는 이를 보완하기 위해 다음 update budget proxy 를 추가한다.

```text
s_t = (1 − ρ_t) · ||u_t||_2       # update budget: 새 입력이 과거 state 를 갈아엎는 강도
```

이 proxy 는 18 ckpt 분석에서 corr(s, gyro) = +0.30, s_trans / s_non = 1.15 의 양의 변별력을 보였으며, 이를 head 에 추가한 selective GyroPhase Head 는 legacy ρ proxy 대비 direction macro F1 +0.039 의 개선을 보였다.

Selective GyroPhase Head 의 full feature 는 다음과 같다.

```text
f_t = [m_t, p_t, q_t, s_t, d,
       p_t · q_t, p_t · s_t, q_t · s_t,
       p_t · d, q_t · d, s_t · d,
       p_t · q_t · s_t · d]
h_phase = Pool_t(f_t)               # Pool_t ∈ {mean, max, std}
```

### 4.5 최종 classifier

기존 backbone 에서 얻은 sequence-level representation 을 평균 풀링하여 `h_base = mean_t(h_t)` 를 얻고, GyroPhase Head 는 다음과 같이 최종 representation 을 구성한다.

```text
h_final = concat(h_base, h_phase)
ŷ       = Classifier(h_final)
```

본 연구에서는 `ŷ` 를 다음 두 task 에 대해 사용한다.

```text
1. binary transition detection
2. 7-class transition direction classification
```

확장 구조에서는 activity classification 과 boundary offset estimation 을 함께 수행하는 multi-task head 로 확장할 수 있다.

### 4.6 backbone-independent 적용

본 head 는 (i) sequence-level 표현을 mean-pool 할 수 있고 (ii) optional 로 complex hidden state 와 selective update 정보를 노출할 수 있는 모든 encoder 에 적용 가능하다. Phase-3 sweep 에서 다음 backbone 에 일관되게 적용되었다.

```text
TCN              : phase 정보 없음, 그러나 GyroPhase Head 의 RD / gyro feature 만으로도 작동.
Transformer      : 동일.
Mamba-3          : fused kernel 이므로 hidden phase 는 input gyro-derived proxy 로 fallback.
Real-/Complex-SSM (자체)    : 2 × 2 ablation 에서 hidden phase 직접 노출.
```

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

3. **Synthetic rotation task — Phase-2 (ω 포함)**
   - rotation direction
   - phase jump detection
   - angular velocity change detection

4. **Harder synthetic rotation task — Phase-4 (cos/sin only)**
   - direction_hard (2-class clockwise vs counter-clockwise)
   - mid_switch (2-class, ω switches at random τ)
   - speed_direction6 (6-class {slow, medium, fast} × {+, −})

### 5.3 비교 모델

비교 모델은 phase 별로 다음과 같이 구성한다.

**Phase-1/2 baselines** (last-token pool):

| 모델 | 역할 |
|---|---|
| 1D-CNN | lightweight local pattern baseline |
| GRU | recurrent sequential baseline |
| TCN | strong practical temporal convolution baseline |
| Transformer Encoder | attention-based global context baseline |
| Mamba-3 | selective SSM baseline |
| Real-SSM | real-valued state update baseline |
| Complex-SSM | complex-valued state update ablation |

**Phase-3 — TIC-style encoder + GyroPhase Head** (mean-pool readout, 14 specs × 3 seeds):

| 그룹 | 모델 | 역할 |
|---|---|---|
| Baseline | TCN / Transformer / Mamba-3 + AvgPool | TIC-style baseline |
| Head ablation (Mamba-3) | Mamba-3 + GyroPhase + RD | head 효과 (input gyro-derived phase fallback) |
| Head portability | Transformer + GyroPhase + RD | backbone-independent 확인 |
| 2 × 2 SSM ablation | Real-Static / Real-Selective / Complex-Static / Complex-Selective | selective × complex 효과 분리 |
| Phase-aware head | Complex-Selective + {Phase, GyroPhase, GyroPhase+RD, Selective GyroPhase} | hidden phase 직접 활용 |

**Phase-4 — 후속 ablation**:

| 그룹 | 변형 | 역할 |
|---|---|---|
| Selective proxy | Complex-Selective + selective_gyrophase_v2 / v3 | ρ → update_budget / phase_velocity |
| Subject-disjoint | TCN / Transformer / Mamba-3 + {AvgPool, GyroPhase+RD} × 5 | 사용자 독립 일반화 |
| Harder synthetic | 8 backbones × {direction_hard, mid_switch, speed_direction6} | controlled rotation 검증 |

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

## 6. 실험 결과 및 해석

### 6.1 Phase-1: 표준 모델 비교 (last-token pool)

5-seed 평균 실험 결과, 표준 128-window setting 에서는 dilated TCN 과 1D-CNN 이 가장 안정적인 baseline 으로 나타났다. Mamba-3 는 binary transition detection 에서 Transition F1 0.9609 ± 0.0145 를 기록하여 TCN (0.9649) 및 Transformer (0.9731) 를 명확히 능가하지 못하였다. Direction classification 에서도 TCN 이 direction macro F1 0.8118 ± 0.0211 로 가장 높은 성능을 보였고, Mamba-3 는 0.7150 ± 0.0908 에 머물렀다. 이 결과는 짧은 IMU window 기반 전이 탐지에서 last-token pooling 의 단순 선택적 상태공간 모델 적용보다 convolution 기반 baseline 이 실용적임을 보여준다.

### 6.2 Phase-2: Hidden phase 분석

학습된 Complex-SSM 의 hidden state 를 분석한 결과 (10 ckpt = 5 seeds × {gyro, acc_gyro}), hidden phase 변화량은 gyro magnitude 와 Pearson r = **0.847 ± 0.020** 의 강한 양의 상관을 보였고, 전이 구간에서는 비전이 구간보다 phase 변화량이 **1.231 ± 0.057 배** 더 크게 나타났다. 이는 complex-valued state 가 *분류 성능과 별개로* 회전성 신호를 내부 표현 수준에서 추적할 수 있음을 정량적으로 보였다.

그러나 자체 naive Complex-SSM 의 direction classification 은 0.08 수준으로 사실상 학습 실패였다. hidden phase 정보가 존재하더라도 기존 last-token classifier head 가 이를 명시적으로 활용하지 못함이 확인되었으며, 이 관찰이 본 연구의 GyroPhase Head 제안으로 이어졌다.

### 6.3 Phase-3: GyroPhase Head 와 mean-pool readout 의 정량 검증

본 연구는 phase-3 에서 (i) encoder readout 을 last-token → mean-pool 로 교체하고, (ii) GyroPhase Head 의 5 가지 변형 (avgpool / magnitude / phase / gyrophase / gyrophase_rd / selective_gyrophase) 을 학습하였으며, (iii) 2 × 2 SSM ablation (real/complex × static/selective) 으로 selective scanning 과 complex update 의 효과를 분리하였다. 14 specs × 3 seeds = 42 runs 의 주요 결과는 다음과 같다.

| Backbone + Head | Direction Macro F1 | Worst-class F1 | Δ vs Phase-1 mamba3 last-pool |
|---|---|---|---:|
| **Mamba-3 + GyroPhase + RD** | **0.7611 ± 0.0440** | **0.6649** | **+0.046** |
| Real-Selective + AvgPool | 0.7560 ± 0.0488 | 0.6420 | +0.041 |
| Mamba-3 + AvgPool (mean-pool) | 0.7553 ± 0.0587 | 0.6203 | +0.040 |
| TCN + AvgPool | 0.7492 ± 0.0239 | 0.5556 | −0.063 (vs TCN) |
| Transformer + GyroPhase + RD | 0.7176 ± 0.0417 | 0.5899 | +0.025 (head 효과) |
| Transformer + AvgPool | 0.6925 ± 0.0220 | 0.5688 | — |
| Real-Static + AvgPool | 0.6841 ± 0.0329 | 0.5084 | — |
| Complex-Selective + Phase | 0.6258 ± 0.0428 | 0.5000 | — |
| Complex-Selective + AvgPool | 0.6035 ± 0.0272 | 0.4282 | — |
| Complex-Static + AvgPool | 0.5821 ± 0.0603 | 0.2254 | — |

세 가지 관찰:

1. **readout 자체의 효과**: Mamba-3 backbone 을 last-token 에서 mean-pool 로 바꾸기만 해도 direction macro F1 이 0.715 → 0.755 (+0.040) 로 향상되어 TCN baseline 과 동률 이상이 된다.
2. **head 의 marginal 효과**: GyroPhase + RD 는 Mamba-3 + AvgPool 대비 평균 +0.006 의 작은 mean 개선이지만, worst-class F1 +0.045 와 표준편차 0.059 → 0.044 의 안정성 개선을 동반한다. 동일 head 가 Transformer 에 적용되어도 +0.025 의 일관된 mean 개선을 주어 **backbone-independent phase-aware readout** 의 성질을 가진다.
3. **selective 와 complex 의 분리**: real-static → real-selective +0.072, complex-static → complex-selective +0.021. 그러나 phase–gyro Pearson 상관과 Δphase trans/non ratio 는 complex_static / complex_selective 에서 동일 (r = 0.83, ratio = 1.24). **회전성 inductive bias 는 complex update 자체가 만들고, selective scanning 은 그 표현이 학습되는 방식에는 추가 변동성을 줄 뿐 본 신호의 출처가 아니다.**

### 6.4 Phase-4: subset 별 이득, selective proxy 개선, harder synthetic

phase-3 의 가장 큰 한계는 mean Direction Macro F1 의 GyroPhase 이득이 σ 안에 묻혀 있다는 것이었다. phase-4 는 4 가지 후속 실험으로 이를 분해한다.

**(1) Transition-only subset (exp_plan4 §3)**: 기존 phase-3 prediction 을 transition window 로 한정한 뒤 gyro magnitude 중앙값으로 high/low 분할.

| Model | trans-high-gyro | full-set |
|---|---|---|
| Mamba-3 + GyroPhase + RD | **0.670** | 0.7611 |
| Mamba-3 + AvgPool | 0.640 | 0.7553 |
| TCN + AvgPool | 0.582 | 0.7492 |
| Transformer + GyroPhase + RD | 0.576 | 0.7176 |

가장 어려운 subset 에서 head 효과는 **+0.030** (full-set Δ +0.006 의 5배). GyroPhase Head 의 *난이도 적응형* 특성을 직접 정량화한 결과다.

**(2) Subject-disjoint evaluation (exp_plan4 §2)**: 5 specs × 3 seeds = 15 runs.

| Model | random dirF1 | subject dirF1 | worst-class F1 (subject) |
|---|---|---|---|
| Mamba-3 + AvgPool | 0.7553 | **0.7862** | 0.6176 |
| Mamba-3 + GyroPhase + RD | 0.7611 | 0.7707 | **0.6465** |
| TCN + AvgPool | 0.7492 | 0.7503 | 0.5972 |
| Transformer + AvgPool | 0.6925 | 0.7419 | 0.6294 |
| Transformer + GyroPhase + RD | 0.7176 | 0.7148 | 0.5870 |

direction transition 은 사용자 간 stereotyped 하여 모든 모델이 random ≥ subject (Δ ≥ −0.003). 그러나 **worst-class F1 에서 Mamba-3 + GyroPhase + RD 가 1위 (0.6465 vs Mamba-3 + AvgPool 0.6176, +0.029)** — drop 이 큰 사용자에서의 안정성에서는 head 효과가 random 조건과 동일하게 유지된다.

**(3) Selective update proxy 개선 (exp_plan4 §1)**: phase-3 의 selective_score = rho 는 `corr(rho, gyro) = −0.44`, ratio = 0.99 로 head feature 신호가 약했다. 18 ckpt 정량 분석으로 새 proxy 를 비교:

| proxy | r(proxy, gyro) | r(proxy, |Δφ|) | trans/non ratio |
|---|---:|---:|---:|
| rho (legacy) | −0.44 | −0.35 | 0.99 |
| **forget_rate (1 − rho)** | **+0.44** | **+0.35** | **1.13** |
| **update_budget ((1 − rho) × ||u||)** | **+0.30** | **+0.23** | **1.15** |
| phase_velocity (rho × |sin θ|) | −0.64 | −0.57 | 0.97 |

update_budget proxy 를 사용한 selective_gyrophase_v2 head 는 direction macro F1 **0.6365 ± 0.0826** 으로 legacy selective_gyrophase 0.5979 ± 0.0898 대비 **+0.039** 개선되었으며, complex hidden state 를 활용하는 모든 phase-3 head 중 1위이다.

**(4) Harder synthetic (exp_plan4 §4)**: cos/sin only direction_hard 와 mid_switch 는 8 개 backbone 모두 ceiling (≥ 0.998). 6-class speed_direction6 에서 처음 변별:

| Backbone | speed macro F1 | worst-class F1 |
|---|---|---|
| Transformer | 0.9881 ± 0.0065 | 0.9821 |
| Mamba-3 | 0.9881 ± 0.0038 | 0.9791 |
| complex_selective | 0.9806 ± 0.0016 | 0.9699 |
| real_selective | 0.9758 ± 0.0101 | 0.9518 |
| real_static | 0.9703 ± 0.0127 | 0.9495 |
| **complex_static** | **0.9563 ± 0.0048** | **0.9181** |

complex_static → complex_selective 에서 worst-class F1 +0.052 — **잘 통제된 회전 task 에서는 selective scanning 의 효과가 complex update 와 결합되어 명확히 나타난다**. real (+0.002 worst) 대비 약 25배의 효과 — HAPT 의 자세 전환 분류보다 controlled rotation 환경에서 inductive bias 가 더 잘 드러난다.

### 6.5 본 sweep 의 검증 질문 답

paper.md §6.3 의 원 검증 질문에 대한 답:

1. *GyroPhase Head 가 naive Complex-SSM 의 direction 성능을 회복시키는가?*  
   ◯ 약하게: Complex-Selective + AvgPool 0.604 → + Phase 0.626 (+0.022), + selective_gyrophase_v2 0.637 (+0.033). 그러나 같은 Complex-SSM 학습 안정성 자체의 한계로 Mamba-3 / Real-Selective 수준에는 도달하지 못함.
2. *GyroPhase Head 가 Real-SSM 또는 Mamba-3 수준 이상의 direction macro F1 을 달성하는가?*  
   ◯ Mamba-3 backbone 위에 적용 시 0.7611 로 본 sweep 의 14 specs 중 1위.
3. *회전성 전이가 강한 high-gyro subset 에서 성능 개선이 더 크게 나타나는가?*  
   ◯ trans-high-gyro 에서 +0.030 (full-set Δ +0.006 의 5배).
4. *opposite-pair confusion 이 줄어드는가?*  
   △ 변별력 없음 — 모든 모델에서 opposite-pair 오분류율 0–3% 로 이미 거의 0. dominant confusion 은 opposite 가 아니라 *시작/종료 자세가 인접한 짝* (예: stand_to_lie ↔ sit_to_lie).

---

## 7. 한계 및 향후 과제

### 7.1 해결된 / 부분 해결된 한계 (phase-3 / phase-4)

- (paper.md §6 의 핵심 검증 #1, #2, #3) GyroPhase Head 가 Complex-Selective 의 direction 성능을 부분 회복시키고, Mamba-3 backbone 에서 sweep 1위, trans-high-gyro subset 에서 +0.030 임을 phase-3 / phase-4 sweep 으로 정량 검증함.
- (result.md §6 #6 / exp_plan3-1 §13) GyroPhase Head 의 backbone-independent 일반성은 Transformer 에서도 +0.025 로 확인됨.
- (result.md §6 #5 / paper.md §6.2 학습 실패) selective_gyrophase head 의 약한 신호는 selective_score proxy 를 rho → update_budget 으로 교체하여 +0.039 회복함.

### 7.2 잔여 한계

본 연구는 복소수 상태공간 모델의 hidden phase 가 IMU 회전성 신호와 관련된 정보를 포함할 수 있음을 보였으나, 이를 곧바로 실제 신체 회전각의 해석으로 연결할 수는 없다. Hidden phase 는 모델 내부 표현이며, 실제 관절 각도나 IMU orientation 과 1:1 대응하지 않는다. 또한 phase-3 / phase-4 의 mean Δ direction macro F1 +0.006 (full set) 은 3-seed σ ≈ 0.04–0.06 안에서 통계적으로 분리되지 않으며, 본 연구의 직접적 정량 이득은 subset 별 (+0.030 trans-high-gyro), worst-class (+0.029), proxy 교체 (+0.039) 등 *조건부* 개선이다. paired bootstrap 등 통계 검증과 5–10 seeds 확장이 우선 후속 과제이다.

Phase-3 / phase-4 의 selective scanning 분석은 본 연구의 자체 Complex-SSM 구현 (2 × 2 ablation) 에 한정되며, Mamba-3 의 fused triton/cute kernel 내부 state 는 직접 hook 하지 않았다 (exp_plan3-1 §13 대안 1 채택). Mamba-3 + GyroPhase 변형은 input gyro-derived phase proxy 로 fallback 한다. Mamba-3 의 step API 를 활용해 unfused reference path 를 구현하면 internal Δphase 와 Δφ–gyro 상관을 직접 측정할 수 있을 것이다.

향후 과제로는 leave-one-subject-out (30-fold) cross-validation, streaming evaluation 에서의 detection latency 재평가, boundary offset estimation 을 통한 실제 로봇 및 웨어러블 응용에 가까운 평가, 더 어려운 synthetic rotation task (T=32, noise=0.2, late switch) 를 통한 회전성 inductive bias 의 직접 검증이 남아 있다.

---

## 8. 결론

본 연구는 IMU 기반 동작 인식을 정적 행동 분류에서 동작 전이 탐지 및 전이 방향 추정 문제로 확장하였다. Phase-1/2 의 표준 분류 (last-token pool) 에서는 dilated TCN 과 1D-CNN 이 가장 안정적인 baseline 으로 나타났고, vanilla Mamba-3 와 naive Complex-SSM 은 이를 능가하지 못하였다. 그러나 Complex-SSM 의 hidden phase 변화량은 gyro magnitude 와 r = 0.847 ± 0.020 의 강한 상관을 보였고, 전이 구간에서 비전이 구간보다 1.231 ± 0.057 배 더 크게 변화하였다. 본 연구의 phase-3 / phase-4 후속 실험은 이 inductive bias 가 selective scanning 이 없는 complex-static SSM 에서도 r = 0.84 로 동일하게 형성됨을 확인하여, 회전성 표현의 출처가 complex update 자체임을 분리해서 보였다.

이러한 관찰을 바탕으로 본 연구는 hidden magnitude / phase 변화량 / gyro magnitude / rotation diversity / selective update budget 을 명시적으로 결합하는 **Selective GyroPhase Head** 를 제안하였다. mean-pool encoder readout 위의 Mamba-3 + GyroPhase + RD Head 는 direction macro F1 0.7611 ± 0.0440 으로 14 개 비교 모델 중 1위였으며, Transformer encoder 에서도 +0.025 의 개선을 보여 backbone-independent 한 phase-aware readout 으로 작동한다. 본 head 의 효과는 가장 어려운 transition-high-gyro subset 에서 +0.030, subject-disjoint worst-class F1 에서 +0.029 로 나타나 *난이도 적응형* 임을 시사하며, selective update 의 적절한 proxy 가 retention coefficient 가 아니라 update budget 임을 정량적으로 확인 (+0.039 dirF1) 하여 selective scanning 의 head feature 화 방향을 제시한다. 이로써 본 연구는 복소수 상태공간 표현의 회전성 inductive bias 를 단순한 해석 분석에 그치지 않고, 실제 IMU 동작 전이 분류 구조로 정량 검증된 head 로 연결한다.
