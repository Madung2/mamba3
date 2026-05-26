 IMU 동작 전이 탐지에서 Mamba-3 복소수 상태 업데이트의 회전성 표현 가능성 분석 — 실질 구현 계획서 v2

## 0. 핵심 논지

본 연구의 중심 가설은 다음과 같이 정리한다.

> **Mamba-3의 complex-valued state update는 hidden state 공간에서 크기 변화와 위상 변화를 함께 표현할 수 있으므로, IMU의 각속도·자세 전환처럼 회전성 또는 주기성이 포함된 시계열을 표현하는 데 유리한 inductive bias를 제공할 가능성이 있다.**

단, 이 문장은 다음처럼 방어적으로 해석해야 한다.

- “Mamba-3가 실제 신체 회전을 직접 이해한다”는 뜻이 아니다.
- “복소수 hidden state의 phase가 실제 관절 각도와 1:1 대응된다”는 뜻도 아니다.
- 더 안전한 주장은 다음이다.

> **복소수 상태 업데이트는 hidden state에서 크기와 위상 변화를 함께 표현할 수 있으므로, 회전성·주기성·방향성 변화가 포함된 IMU 시계열을 표현하는 데 구조적 편향을 제공할 수 있다. 본 연구는 이 가능성을 실험적으로 검토한다.**

따라서 논문의 목적은 **Mamba-3가 모든 모델보다 좋다는 주장**이 아니라, **IMU 동작 전이 탐지에서 complex-valued state update가 회전성 시계열 표현에 어떤 가능성과 한계를 가지는지 분석하는 것**이다.

---

## 1. 현재 실험 1 요약

현재 완료된 15-run sweep 결과는 다음과 같다.

### 실험 1: Binary Transition Detection, acc+gyro 기준

| Model | Acc | Macro F1 | Trans P | Trans R | Trans F1 | ms/win |
|---|---:|---:|---:|---:|---:|---:|
| 1D-CNN | 0.9957 | 0.9769 | 0.9383 | 0.9744 | 0.9560 | 0.0018 |
| GRU | 0.9951 | 0.9734 | 0.9375 | 0.9615 | 0.9494 | 0.0025 |
| TCN | 0.9963 | 0.9801 | 0.9500 | 0.9744 | 0.9620 | 0.0058 |
| Transformer | 0.9970 | 0.9831 | 0.9740 | 0.9615 | 0.9677 | 0.0039 |
| Mamba-3 | 0.9970 | 0.9829 | 0.9867 | 0.9487 | 0.9673 | 0.0121 |

### 현재 결과 해석

- Binary transition detection에서는 전체 모델 성능이 이미 높다.
- Mamba-3는 Transformer와 거의 동일한 Transition F1을 보였다.
- Mamba-3는 Transition Precision이 가장 높다.
- 하지만 Mamba-3는 ms/window가 가장 높아, 단순 경량성 우위는 주장하기 어렵다.
- 따라서 이후 실험은 **“Mamba-3가 더 빠르다”**가 아니라 **“복소수 상태 업데이트가 회전성/방향성 전이 표현에 유리한가”**를 검증하는 방향으로 설계한다.

---

## 2. 전체 실험 방향

기존 실험은 다음 문제를 다뤘다.

```text
IMU window → transition / non-transition
```

이 문제는 너무 쉽고, 1D-CNN/TCN/Transformer도 충분히 강하다. 따라서 Mamba-3의 필요성을 보이려면 문제를 다음처럼 확장해야 한다.

```text
1. 회전성 센서 신호가 있는 조건에서 성능이 좋아지는가?
2. 전이 여부뿐 아니라 전이 방향을 잘 구분하는가?
3. real-valued SSM보다 complex-valued SSM이 회전성 전이에 유리한가?
4. 실제 IMU만으로 불명확하면 synthetic rotation task에서 구조적 이점을 검증할 수 있는가?
```

이에 따라 다음 4개 실험을 핵심 실험으로 구성한다.

| 번호 | 실험 | 목적 |
|---:|---|---|
| 실험 1 | Sensor Ablation | acc only / gyro only / acc+gyro 조건에서 Mamba-3의 특성 확인 |
| 실험 2 | Transition Direction Classification | 단순 전이 여부가 아니라 전이 방향 분류 |
| 실험 3 | Real SSM vs Complex SSM Ablation | 복소수 상태 업데이트 자체의 기여도 검증 |
| 실험 4 | Synthetic Rotation Task | 회전성 시계열에서 complex update의 구조적 이점 검증 |

추가로 모든 실험에 대해 가능한 경우 다음 분석을 붙인다.

```text
- window length sweep
- confusion matrix
- false positive / false negative 분석
- hidden phase 변화량과 gyro magnitude 상관 분석
```

---

## 3. 공통 데이터셋 및 입력 설정

### 3.1 데이터셋

주 데이터셋은 다음을 사용한다.

```text
UCI Smartphone-Based Recognition of Human Activities and Postural Transitions
```

### 3.2 기본 입력

```text
Input shape = [batch_size, seq_len, channels]
```

기본 설정:

| 항목 | 값 |
|---|---|
| Sampling rate | 50Hz |
| Default window | 2.56초 |
| Default seq_len | 128 |
| Overlap | 50% |
| Channels | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z |

### 3.3 Sensor channel set

실험별 입력 채널은 다음처럼 분리한다.

| 조건 | 입력 채널 | channel 수 |
|---|---|---:|
| acc_only | acc_x, acc_y, acc_z | 3 |
| gyro_only | gyro_x, gyro_y, gyro_z | 3 |
| acc_gyro | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z | 6 |

### 3.4 Window length sweep

모든 실험에서 가능하면 아래 window length를 비교한다.

| 이름 | 길이 | timestep |
|---|---:|---:|
| short | 1.28초 | 64 |
| default | 2.56초 | 128 |
| long | 5.12초 | 256 |
| extra_long | 10.24초 | 512 |

초기 구현은 `128`로 고정하고, 이후 `64/256/512`를 추가한다.

---

## 4. 공통 모델 목록

기본 비교 모델은 다음과 같다.

| 모델 | 역할 |
|---|---|
| 1D-CNN | 가장 가벼운 local pattern baseline |
| GRU 또는 BiLSTM | 전통적인 sequential state baseline |
| TCN | 강한 practical 시계열 baseline |
| Transformer Encoder | attention 기반 global context baseline |
| Real-valued SSM / Mamba-2-like | real-valued state update baseline |
| Complex-valued SSM / Mamba-3-like | complex state update 가설 검증 모델 |
| Mamba-3 | 최종 제안 모델 또는 Mamba-3 기반 모델 |

중요한 점은 TCN을 약한 baseline으로 취급하지 않는 것이다. TCN은 짧은 IMU window에서 매우 강할 수 있다. 본 연구는 TCN보다 무조건 빠르거나 좋다고 주장하지 않는다.

---

## 5. 공통 평가 지표

### 5.1 Classification metric

| 지표 | 설명 |
|---|---|
| Accuracy | 전체 정확도 |
| Macro F1 | 클래스 불균형을 고려한 평균 F1 |
| Weighted F1 | 클래스 수가 불균형할 때 보조 지표 |
| Per-class F1 | 각 transition class별 F1 |
| Confusion Matrix | 어떤 전이 방향을 혼동하는지 분석 |

### 5.2 Transition-specific metric

| 지표 | 설명 |
|---|---|
| Transition Precision | 전이라고 판단한 것 중 실제 전이 비율 |
| Transition Recall | 실제 전이를 얼마나 놓치지 않았는지 |
| Transition F1 | 전이 탐지 핵심 지표 |
| Direction Macro F1 | 전이 방향 분류 성능 |
| Transition-class Macro F1 | transition class만 대상으로 한 평균 F1 |

### 5.3 Cost metric

| 지표 | 설명 |
|---|---|
| Params | 모델 파라미터 수 |
| MACs/FLOPs | 계산량 |
| ms/window | window 1개 추론 시간 |
| GPU memory | 학습/추론 시 메모리 사용량 |

### 5.4 회전성 분석 metric

| 지표 | 설명 |
|---|---|
| gyro magnitude | `sqrt(gx^2 + gy^2 + gz^2)` |
| hidden phase | `atan2(Im(h), Re(h))` |
| delta phase | `phase_t - phase_{t-1}` |
| phase-gyro correlation | hidden phase 변화와 gyro magnitude의 상관 |
| event-aligned phase plot | 전이 시점 기준 phase 변화 시각화 |

---

# 실험 1. Sensor Ablation

## 1.1 목적

Mamba-3의 complex-valued state update가 회전성 신호에 유리하다는 논지를 검증하려면, `gyro` 신호가 포함된 조건에서 Mamba-3의 상대적 특성이 달라지는지 봐야 한다.

핵심 질문:

```text
Q1. Mamba-3는 acc_only보다 gyro_only 또는 acc+gyro 조건에서 상대적으로 더 강한가?
Q2. Mamba-3의 높은 precision 특성이 gyro 기반 전이 탐지에서 더 뚜렷해지는가?
Q3. 회전성 신호가 빠졌을 때 Mamba-3의 장점이 약해지는가?
```

## 1.2 실험 조건

| 조건 | 입력 |
|---|---|
| acc_only | acc_x, acc_y, acc_z |
| gyro_only | gyro_x, gyro_y, gyro_z |
| acc_gyro | acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z |

각 조건에서 동일한 모델을 반복 학습한다.

```text
models = [1D-CNN, GRU, TCN, Transformer, Mamba-3]
seeds = [0, 1, 2, ..., 14]
```

## 1.3 구현 체크리스트

### Dataset config

```yaml
dataset:
  name: uci_hapt
  task: binary_transition
  seq_len: 128
  overlap: 0.5
  sensor_mode: acc_only   # acc_only | gyro_only | acc_gyro
```

### 실행 예시

```bash
python train.py --task binary_transition --sensor_mode acc_only  --model tcn --seed 0
python train.py --task binary_transition --sensor_mode gyro_only --model tcn --seed 0
python train.py --task binary_transition --sensor_mode acc_gyro  --model tcn --seed 0

python train.py --task binary_transition --sensor_mode acc_only  --model mamba3 --seed 0
python train.py --task binary_transition --sensor_mode gyro_only --model mamba3 --seed 0
python train.py --task binary_transition --sensor_mode acc_gyro  --model mamba3 --seed 0
```

### 결과 파일 구조

```text
results/
  sensor_ablation/
    acc_only/
      cnn.csv
      gru.csv
      tcn.csv
      transformer.csv
      mamba3.csv
    gyro_only/
      ...
    acc_gyro/
      ...
```

## 1.4 결과 표 템플릿

| Sensor | Model | Acc | Macro F1 | Trans P | Trans R | Trans F1 | ms/win |
|---|---|---:|---:|---:|---:|---:|---:|
| acc_only | TCN | - | - | - | - | - | - |
| acc_only | Mamba-3 | - | - | - | - | - | - |
| gyro_only | TCN | - | - | - | - | - | - |
| gyro_only | Mamba-3 | - | - | - | - | - | - |
| acc_gyro | TCN | - | - | - | - | - | - |
| acc_gyro | Mamba-3 | - | - | - | - | - | - |

## 1.5 기대되는 해석 패턴

### 좋은 결과 패턴

```text
acc_only에서는 Mamba-3 이점이 약함
gyro_only 또는 acc+gyro에서는 Mamba-3의 Precision/F1이 상대적으로 좋아짐
```

논문 문장:

> Mamba-3의 성능 이점은 선형 가속도만 사용한 조건보다 각속도 신호가 포함된 조건에서 더 뚜렷하게 나타났다. 이는 complex-valued state update가 회전성 또는 위상성 변화가 포함된 IMU 시계열을 표현하는 데 유리한 inductive bias를 제공할 가능성을 시사한다.

### 애매한 결과 패턴

```text
TCN이 모든 sensor 조건에서 가장 좋음
Mamba-3는 precision은 높지만 recall이 낮음
```

논문 문장:

> 짧은 IMU window 기반 binary transition detection에서는 TCN이 가장 실용적인 baseline으로 나타났다. 다만 Mamba-3는 gyro 포함 조건에서 높은 precision을 보여, 전이 오탐을 줄이는 방향의 모델 특성을 확인할 수 있었다.

---

# 실험 2. Transition Direction Classification

## 2.1 목적

Binary transition detection은 다음처럼 단순하다.

```text
transition / non-transition
```

그러나 실제 로봇·재활·고령자 케어에서는 단순히 전이가 발생했다는 사실보다, **어떤 상태에서 어떤 상태로 바뀌는지**가 더 중요하다.

예:

```text
sit-to-stand: 사용자가 일어나려는 상황
stand-to-sit: 사용자가 앉으려는 상황
lie-to-stand: 누운 상태에서 일어나는 상황
stand-to-lie: 쓰러짐 또는 눕는 상황
```

따라서 이 실험에서는 전이 여부가 아니라 전이 방향을 분류한다.

## 2.2 라벨 정의

### 7-class direction task

| Class index | Label |
|---:|---|
| 0 | non-transition |
| 1 | stand-to-sit |
| 2 | sit-to-stand |
| 3 | sit-to-lie |
| 4 | lie-to-sit |
| 5 | stand-to-lie |
| 6 | lie-to-stand |

### 주의점

`non-transition` class가 매우 클 수 있으므로 전체 Accuracy만 보면 안 된다. 반드시 다음 지표를 함께 사용한다.

```text
Macro F1
Transition-class Macro F1
Per-transition F1
Confusion Matrix
```

## 2.3 구현 체크리스트

### Dataset label mapping

```python
LABEL_MAP_DIRECTION = {
    "WALKING": 0,
    "WALKING_UPSTAIRS": 0,
    "WALKING_DOWNSTAIRS": 0,
    "SITTING": 0,
    "STANDING": 0,
    "LAYING": 0,

    "STAND_TO_SIT": 1,
    "SIT_TO_STAND": 2,
    "SIT_TO_LIE": 3,
    "LIE_TO_SIT": 4,
    "STAND_TO_LIE": 5,
    "LIE_TO_STAND": 6,
}
```

### Training config

```yaml
task: direction_classification
num_classes: 7
loss: cross_entropy
class_weight: balanced
metric_primary: transition_class_macro_f1
```

### 실행 예시

```bash
python train.py --task direction_classification --sensor_mode acc_gyro --model tcn --seed 0
python train.py --task direction_classification --sensor_mode acc_gyro --model transformer --seed 0
python train.py --task direction_classification --sensor_mode acc_gyro --model mamba3 --seed 0
```

## 2.4 결과 표 템플릿

| Model | Acc | Macro F1 | Non-trans F1 | Direction Macro F1 | Worst Class F1 | ms/win |
|---|---:|---:|---:|---:|---:|---:|
| 1D-CNN | - | - | - | - | - | - |
| GRU | - | - | - | - | - | - |
| TCN | - | - | - | - | - | - |
| Transformer | - | - | - | - | - | - |
| Mamba-3 | - | - | - | - | - | - |

여기서 `Direction Macro F1`은 class 1~6만 평균낸다.

```python
direction_macro_f1 = mean(f1[class_id] for class_id in [1,2,3,4,5,6])
```

## 2.5 Confusion Matrix 분석

전이 방향 분류에서는 confusion matrix가 중요하다.

특히 다음 혼동을 확인한다.

```text
sit-to-stand ↔ stand-to-sit
sit-to-lie ↔ lie-to-sit
stand-to-lie ↔ lie-to-stand
standing/sitting/lying ↔ transition
```

## 2.6 기대되는 해석 패턴

### 좋은 결과 패턴

```text
Mamba-3가 binary task에서는 Transformer와 유사했지만,
direction classification에서는 transition-class macro F1이 더 높음
```

논문 문장:

> 단순 전이 여부 탐지에서는 모델 간 차이가 크지 않았으나, 전이 방향 분류에서는 Mamba-3가 더 안정적인 transition-class macro F1을 보였다. 이는 complex-valued state update가 상태 변화의 방향성을 표현하는 데 유리한 구조적 편향을 제공할 가능성을 보여준다.

### 애매한 결과 패턴

```text
TCN 또는 Transformer가 direction F1 1등
Mamba-3는 특정 class에서만 좋음
```

논문 문장:

> 전이 방향 분류 전체에서는 TCN/Transformer가 강한 성능을 보였으나, Mamba-3는 특정 자세 전이 class에서 높은 precision을 보였다. 이는 Mamba-3가 모든 전이 방향에 일관적으로 우수하다기보다, 특정 회전성 변화가 강한 전이에서 선택적으로 이점을 가질 수 있음을 시사한다.

---

# 실험 3. Real-valued SSM vs Complex-valued SSM Ablation

## 3.1 목적

Mamba-3의 핵심 간지 포인트는 **complex-valued state update**다. 따라서 Mamba-3와 TCN만 비교하면 부족하다.

진짜로 검증해야 할 질문은 다음이다.

```text
Q. real-valued state update보다 complex-valued state update가 회전성 IMU 전이에 유리한가?
```

따라서 비교군은 다음처럼 잡는다.

| 모델 | 목적 |
|---|---|
| Real-SSM | real-valued state update baseline |
| Complex-SSM | complex-valued state update 가설 검증 |
| Mamba-2-like | 기존 Mamba 계열 baseline |
| Mamba-3-like | complex update 포함 모델 |
| TCN | 강한 practical baseline |
| Transformer | global context baseline |

## 3.2 최소 구현 전략

Mamba-3 공식 구현이 환경에서 부담스럽다면, 먼저 **Mamba-3-inspired Complex SSM block**을 작은 형태로 구현한다.

핵심은 real state와 complex state를 동일한 파라미터 규모에 가깝게 맞춰 비교하는 것이다.

## 3.3 Real SSM block 개념

```text
h_t = a_t * h_{t-1} + b_t * x_t
y_t = c_t * h_t
```

여기서 `a_t`, `b_t`, `c_t`는 입력에 따라 생성되는 data-dependent parameter로 둘 수 있다.

간단한 PyTorch 형태:

```python
class RealSSMBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_state)
        self.a_proj = nn.Linear(d_model, d_state)
        self.b_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t]
            a = torch.sigmoid(self.a_proj(xt))
            b = self.b_proj(xt)
            u = self.in_proj(xt)
            h = a * h + b * u
            outputs.append(self.out_proj(h))

        return torch.stack(outputs, dim=1)
```

## 3.4 Complex SSM block 개념

복소수 업데이트는 다음 아이디어를 따른다.

```text
z_t = ρ_t · exp(i θ_t) · z_{t-1} + B_t x_t
```

여기서:

| 기호 | 의미 |
|---|---|
| `ρ_t` | hidden state magnitude decay/growth |
| `θ_t` | hidden state phase rotation |
| `z_t` | complex hidden state |
| `B_t x_t` | 입력 주입 |

실제 구현에서는 PyTorch complex tensor를 직접 써도 되지만, 안정성을 위해 real/imag를 분리해 구현하는 것이 좋다.

```text
z_t = real_t + i imag_t
```

회전 행렬로 쓰면 다음과 같다.

```text
real'_t = ρ_t * (cos θ_t * real_{t-1} - sin θ_t * imag_{t-1})
imag'_t = ρ_t * (sin θ_t * real_{t-1} + cos θ_t * imag_{t-1})
```

간단한 PyTorch 형태:

```python
class ComplexSSMBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_state * 2)  # input to real/imag
        self.rho_proj = nn.Linear(d_model, d_state)
        self.theta_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state * 2, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        real = torch.zeros(B, self.d_state, device=x.device)
        imag = torch.zeros(B, self.d_state, device=x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t]

            rho = torch.sigmoid(self.rho_proj(xt))
            theta = torch.tanh(self.theta_proj(xt)) * math.pi

            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            real_rot = rho * (cos_t * real - sin_t * imag)
            imag_rot = rho * (sin_t * real + cos_t * imag)

            u = self.in_proj(xt)
            u_real, u_imag = u.chunk(2, dim=-1)

            real = real_rot + u_real
            imag = imag_rot + u_imag

            h = torch.cat([real, imag], dim=-1)
            outputs.append(self.out_proj(h))

        return torch.stack(outputs, dim=1)
```

이 구현은 Mamba-3 자체를 완전히 재현한 것은 아니다. 하지만 **real-valued update와 complex-valued update의 차이를 통제된 조건에서 비교하는 ablation**으로 사용할 수 있다.

## 3.5 비교를 공정하게 만드는 조건

| 조건 | 설명 |
|---|---|
| 동일 encoder depth | block 수를 동일하게 |
| 유사 parameter count | d_state 조절 |
| 동일 optimizer | AdamW 등 동일 |
| 동일 epoch | 학습량 동일 |
| 동일 seed | 15-run sweep |
| 동일 input | sensor mode 동일 |
| 동일 task | binary/direction 모두 비교 |

## 3.6 실험 조합

최소 조합:

```text
task = direction_classification
sensor_mode = gyro_only, acc_gyro
models = real_ssm, complex_ssm, tcn, transformer, mamba3
seq_len = 128
seeds = 15
```

확장 조합:

```text
task = direction_classification
sensor_mode = acc_only, gyro_only, acc_gyro
models = real_ssm, complex_ssm, mamba2_like, mamba3_like
seq_len = 64, 128, 256, 512
```

## 3.7 결과 표 템플릿

| Model | State Type | Sensor | Direction Macro F1 | Trans-class F1 | Params | ms/win |
|---|---|---|---:|---:|---:|---:|
| Real-SSM | real | gyro_only | - | - | - | - |
| Complex-SSM | complex | gyro_only | - | - | - | - |
| Real-SSM | real | acc_gyro | - | - | - | - |
| Complex-SSM | complex | acc_gyro | - | - | - | - |
| Mamba-3 | complex-like | acc_gyro | - | - | - | - |

## 3.8 기대되는 해석 패턴

### 좋은 결과 패턴

```text
Complex-SSM > Real-SSM
특히 gyro_only 또는 direction classification에서 차이가 큼
```

논문 문장:

> 동일한 SSM 계열 구조 내에서 real-valued update와 complex-valued update를 비교한 결과, complex update는 gyro 기반 전이 방향 분류에서 더 높은 transition-class macro F1을 보였다. 이는 복소수 상태 업데이트가 회전성 시계열 표현에 유리한 inductive bias를 제공할 수 있음을 시사한다.

### 애매한 결과 패턴

```text
Complex-SSM과 Real-SSM 차이가 거의 없음
```

논문 문장:

> 본 설정에서는 complex-valued update의 명확한 성능 우위가 관찰되지는 않았다. 이는 UCI HAPT의 자세 전이 window가 비교적 짧고, 전이 라벨이 window 단위로 제공되어 phase-level dynamics를 충분히 드러내지 못했기 때문일 수 있다. 향후 더 긴 연속 IMU 시퀀스와 정밀한 boundary label 기반 검증이 필요하다.

---

# 실험 4. Synthetic Rotation Task

## 4.1 목적

실제 IMU 데이터는 사람별 움직임 차이, 센서 위치, 라벨링 방식, 노이즈가 섞여 있어 복소수 상태 업데이트의 장점을 직접 해석하기 어렵다.

따라서 synthetic rotation task를 만든다.

목적은 다음이다.

```text
복소수 상태 업데이트가 회전성/위상성 시계열을 표현하는 데 real-valued update보다 유리한지 통제된 조건에서 확인한다.
```

## 4.2 Synthetic data 생성

### 기본 회전 시계열

```text
θ_t = θ_0 + ω t
x_t = cos(θ_t)
y_t = sin(θ_t)
```

입력:

```text
X_t = [cos(θ_t), sin(θ_t), ω_t]
```

noise 추가:

```text
X_t = X_t + ε,    ε ~ N(0, σ²)
```

### 회전 방향 분류 task

| Label | 조건 |
|---|---|
| clockwise | ω < 0 |
| counter-clockwise | ω > 0 |

### phase jump detection task

특정 시점 `τ`에서 phase jump를 만든다.

```text
θ_t = θ_t + Δθ, if t >= τ
```

라벨:

```text
transition = 1 if phase jump occurs in window
transition = 0 otherwise
```

### angular velocity change task

특정 시점에서 회전 속도가 바뀐다.

```text
ω_t = ω_1, t < τ
ω_t = ω_2, t >= τ
```

라벨:

```text
speed_change = 1 if angular velocity changes
```

## 4.3 구현 예시

```python
def generate_rotation_sequence(
    seq_len=128,
    noise_std=0.05,
    task="direction",
):
    theta0 = np.random.uniform(-np.pi, np.pi)
    omega = np.random.uniform(0.02, 0.20)
    direction = np.random.choice([-1, 1])
    omega = omega * direction

    t = np.arange(seq_len)
    theta = theta0 + omega * t

    if task == "phase_jump":
        has_jump = np.random.rand() < 0.5
        if has_jump:
            tau = np.random.randint(seq_len // 4, seq_len * 3 // 4)
            delta = np.random.uniform(np.pi / 4, np.pi)
            theta[t >= tau] += delta
        label = int(has_jump)

    elif task == "direction":
        label = 1 if direction > 0 else 0

    x = np.stack([
        np.cos(theta),
        np.sin(theta),
        np.full_like(theta, omega),
    ], axis=-1)

    x += np.random.normal(0, noise_std, size=x.shape)

    return x.astype(np.float32), label
```

## 4.4 실험 조건

| 조건 | 값 |
|---|---|
| seq_len | 64, 128, 256, 512 |
| noise_std | 0.00, 0.05, 0.10, 0.20 |
| train samples | 50k |
| val samples | 10k |
| test samples | 10k |
| models | Real-SSM, Complex-SSM, TCN, Transformer, Mamba-3-like |

## 4.5 Task 구성

### Task A. Rotation direction classification

```text
Input: [cos θ_t, sin θ_t, ω_t]
Output: clockwise / counter-clockwise
```

### Task B. Phase jump detection

```text
Input: [cos θ_t, sin θ_t, ω_t]
Output: phase jump / normal
```

### Task C. Angular velocity change detection

```text
Input: [cos θ_t, sin θ_t, ω_t]
Output: speed changed / unchanged
```

## 4.6 결과 표 템플릿

| Task | Model | seq_len | noise | Acc | Macro F1 | Params | ms/win |
|---|---|---:|---:|---:|---:|---:|---:|
| direction | Real-SSM | 128 | 0.05 | - | - | - | - |
| direction | Complex-SSM | 128 | 0.05 | - | - | - | - |
| phase_jump | Real-SSM | 128 | 0.05 | - | - | - | - |
| phase_jump | Complex-SSM | 128 | 0.05 | - | - | - | - |

## 4.7 기대되는 해석 패턴

### 좋은 결과 패턴

```text
Synthetic rotation task에서 Complex-SSM이 Real-SSM보다 안정적으로 높음
특히 noise가 커지거나 seq_len이 길어질 때 차이가 유지됨
```

논문 문장:

> Synthetic rotation sequence에서 complex-valued update는 real-valued update보다 회전 방향 및 phase jump 탐지에서 더 안정적인 성능을 보였다. 이는 복소수 상태 업데이트가 회전성 시계열을 표현하는 데 구조적 이점을 가질 수 있음을 보여준다. 실제 IMU 실험에서는 이러한 가설을 자세 전이 데이터에 적용하여 검토하였다.

### 애매한 결과 패턴

```text
Synthetic에서는 Complex-SSM이 좋지만 실제 IMU에서는 차이가 약함
```

논문 문장:

> 통제된 synthetic rotation task에서는 complex update의 이점이 관찰되었으나, 실제 IMU 데이터에서는 센서 노이즈, 라벨 granularity, 사람별 움직임 차이로 인해 그 효과가 약화되었다. 이는 complex-valued state update의 구조적 가능성과 실제 wearable sensing 환경 사이의 gap을 보여준다.

이 해석은 오히려 석사논문에 좋다. 성공/실패를 모두 분석할 수 있기 때문이다.

---

# 공통 분석. Hidden Phase 변화와 Gyro Magnitude 관계

이 분석은 별도 실험이라기보다, 실험 2~4에 붙이는 해석 분석이다.

## A.1 목적

Mamba-3 또는 Complex-SSM의 hidden state가 complex 형태라면, hidden state의 phase를 계산할 수 있다.

```text
phase_t = atan2(Im(h_t), Re(h_t))
```

이 phase 변화가 전이 구간 또는 gyro magnitude와 함께 증가하는지 본다.

## A.2 계산 방법

```python
phase = torch.atan2(hidden_imag, hidden_real)
delta_phase = wrap_to_pi(phase[:, 1:] - phase[:, :-1])
delta_phase_mag = delta_phase.abs().mean(dim=-1)

gyro_mag = torch.sqrt(gx**2 + gy**2 + gz**2)
```

## A.3 분석 지표

| 지표 | 설명 |
|---|---|
| mean delta phase in transition | 전이 구간에서 phase 변화량 |
| mean delta phase in non-transition | 비전이 구간에서 phase 변화량 |
| phase-gyro correlation | gyro magnitude와 delta phase의 상관 |
| event-aligned plot | 전이 시점을 0으로 정렬한 phase 변화 시각화 |

## A.4 그림 구성

### Figure 1. Event-aligned signal plot

```text
x-axis: time relative to transition center
y-axis 1: gyro magnitude
y-axis 2: hidden delta phase
background: transition interval
```

### Figure 2. Transition vs Non-transition phase distribution

```text
boxplot:
- non-transition delta phase
- transition delta phase
```

## A.5 논문 문장 예시

좋은 결과일 때:

> 전이 구간에서 complex hidden state의 phase 변화량이 비전이 구간보다 크게 나타났으며, gyro magnitude와도 양의 상관을 보였다. 이는 모델의 complex state가 각속도 기반 회전성 변화에 반응하고 있음을 시사한다.

주의할 점:

> hidden phase가 실제 신체 각도와 직접 대응된다고 해석해서는 안 된다. 본 분석은 hidden state의 위상 변화가 전이 구간 및 각속도 변화와 함께 증가하는 경향이 있는지를 확인하는 해석적 분석이다.

---

# 구조 강화. Multi-task Head 구현

실험 결과가 조금 미진해도 논문 구조를 강화하려면 multi-task head를 붙이는 것이 좋다.

## M.1 목적

단일 binary transition detection은 너무 쉽다. 따라서 하나의 shared encoder에서 다음을 동시에 예측하도록 만든다.

```text
1. activity class
2. transition 여부
3. transition direction
```

## M.2 구조

```text
IMU sequence
  ↓
Sensor embedding
  ↓
Shared Encoder
  ↓
Multi-task Heads
    ├─ Activity classification head
    ├─ Binary transition head
    └─ Direction classification head
```

## M.3 출력 정의

| Head | 출력 |
|---|---|
| Activity head | walking, sitting, standing, lying, ... |
| Transition head | transition / non-transition |
| Direction head | stand-to-sit, sit-to-stand, ... |

## M.4 Loss

```text
L_total = L_activity + λ1 L_transition + λ2 L_direction
```

초기값:

```text
λ1 = 1.0
λ2 = 1.0
```

Direction loss는 transition sample에 대해서만 계산한다.

```python
loss_activity = ce(activity_logits, activity_label)
loss_transition = ce(transition_logits, transition_label)

mask = transition_label == 1
if mask.sum() > 0:
    loss_direction = ce(direction_logits[mask], direction_label[mask])
else:
    loss_direction = 0.0

loss = loss_activity + lambda_t * loss_transition + lambda_d * loss_direction
```

## M.5 비교 실험

| 비교 | 목적 |
|---|---|
| single-task binary | 전이 여부만 예측 |
| single-task direction | 전이 방향만 예측 |
| multi-task | activity + transition + direction 동시 예측 |

## M.6 해석

좋은 결과:

> Multi-task 학습은 activity context와 transition direction 정보를 함께 학습하게 하여, 단일 direction classification보다 transition-class macro F1을 개선하였다.

애매한 결과:

> Multi-task 구조는 전체 성능을 크게 개선하지는 않았지만, transition direction 분류에서 특정 class의 혼동을 줄이는 효과를 보였다.

---

# 실행 우선순위

## 1단계: Sensor Ablation

가장 먼저 실행한다.

```text
task = binary_transition
sensor_mode = acc_only, gyro_only, acc_gyro
models = 1D-CNN, TCN, Transformer, Mamba-3
seq_len = 128
seeds = 15
```

목표:

```text
gyro가 포함되었을 때 Mamba-3의 precision/F1 특성이 달라지는지 확인
```

---

## 2단계: Direction Classification

```text
task = direction_classification
sensor_mode = acc_gyro
models = 1D-CNN, TCN, Transformer, Mamba-3
seq_len = 128
seeds = 15
```

목표:

```text
binary보다 어려운 전이 방향 분류에서 모델별 차이 확인
```

---

## 3단계: Real vs Complex SSM

```text
task = direction_classification
sensor_mode = gyro_only, acc_gyro
models = real_ssm, complex_ssm, mamba3
seq_len = 128
seeds = 15
```

목표:

```text
complex update 자체의 기여도 확인
```

---

## 4단계: Synthetic Rotation Task

```text
task = rotation_direction, phase_jump
models = real_ssm, complex_ssm, tcn, transformer
seq_len = 64, 128, 256, 512
noise = 0.00, 0.05, 0.10, 0.20
```

목표:

```text
회전성 시계열에서 complex update의 inductive bias를 통제된 조건에서 확인
```

---

## 5단계: Hidden Phase Analysis

```text
target model = complex_ssm or mamba3
target task = direction_classification
sensor_mode = gyro_only, acc_gyro
```

목표:

```text
hidden phase 변화량이 전이 구간/gyro magnitude와 관련되는지 분석
```

---

# 최종 논문 주장 구조

실험 결과에 따라 다음 중 하나로 주장한다.

## 결과가 잘 나왔을 때

> 본 연구는 IMU 동작 전이 탐지에서 Mamba-3 기반 complex-valued state update가 회전성 신호가 포함된 조건, 특히 gyro-only 및 transition direction classification에서 강점을 보임을 확인하였다. 또한 synthetic rotation task에서 complex update가 real-valued update보다 위상 변화 탐지에 안정적인 성능을 보여, 복소수 상태공간 모델이 회전성 시계열 표현에 유리한 inductive bias를 제공할 가능성을 보였다.

## 결과가 중간일 때

> Mamba-3는 binary transition detection에서 Transformer와 유사한 성능을 보였고, 높은 precision을 통해 전이 오탐을 줄이는 경향을 보였다. 다만 짧은 IMU window에서는 TCN이 강한 실용 baseline으로 나타났으며, complex-valued update의 이점은 gyro 기반 조건과 전이 방향 분류에서 제한적으로 관찰되었다.

## 결과가 미진할 때

> 본 연구는 Mamba-3가 모든 조건에서 기존 모델보다 우수함을 보이지는 못했으나, IMU 동작 전이 문제를 회전성 상태 변화 표현 관점에서 재정의하고, complex-valued state update의 가능성과 한계를 실험적으로 분석하였다. 특히 실제 IMU 데이터와 synthetic rotation task를 함께 사용함으로써, 복소수 상태공간 모델의 구조적 inductive bias가 실제 wearable sensing 문제에서 어떻게 드러나는지 검토하였다.

---

# 2페이지 데모 소논문 구성안

## 제목 후보

### 안전한 제목

```text
IMU 기반 동작 전이 탐지에서 복소수 상태공간 모델의 회전성 표현 가능성 분석
```

### 조금 더 강한 제목

```text
복소수 상태공간 모델을 활용한 IMU 기반 회전성 동작 전이 탐지 연구
```

### Mamba-3를 드러내는 제목

```text
Mamba-3 기반 복소수 상태공간 모델을 활용한 IMU 동작 전이 탐지 연구
```

## 2페이지 구성

```text
1. 초록
2. 서론
3. 관련 연구
4. 제안 방법
5. 실험 설계 및 예비 결과
6. 결론
```

## 2페이지에 넣을 최소 표

### 표 1. Binary transition detection 예비 결과

현재 완료된 실험 1 결과를 넣는다.

### 표 2. 추가 실험 설계

| 실험 | 목적 | 핵심 지표 |
|---|---|---|
| Sensor ablation | gyro 신호 기여도 | Trans F1, Precision |
| Direction classification | 전이 방향성 평가 | Direction Macro F1 |
| Real vs Complex SSM | 복소수 update 기여도 | Transition-class F1 |
| Synthetic rotation | 회전성 inductive bias 검증 | Acc, Macro F1 |

---

# 참고 문헌 및 링크

## Mamba / Mamba-3

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces  
  https://arxiv.org/abs/2312.00752

- Mamba-3: Improved Sequence Modeling using State Space Principles  
  https://arxiv.org/abs/2603.15569

- Mamba-3 OpenReview  
  https://openreview.net/forum?id=HwCvaJOiCj

## Dataset

- UCI Smartphone-Based Recognition of Human Activities and Postural Transitions  
  https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions

- UCI Human Activity Recognition Using Smartphones  
  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

# 바로 해야 할 작업 목록

```text
[ ] sensor_mode 옵션 정리: acc_only / gyro_only / acc_gyro
[ ] direction_classification 라벨 매핑 추가
[ ] direction macro F1 계산 함수 추가
[ ] confusion matrix 저장 추가
[ ] real_ssm block 구현
[ ] complex_ssm block 구현
[ ] synthetic rotation dataset 구현
[ ] hidden state 저장 옵션 추가
[ ] hidden phase 분석 스크립트 추가
[ ] 결과 자동 취합 result.md 업데이트
```

---

# 파일 산출물 구조 추천

```text
project/
  configs/
    binary_acc.yaml
    binary_gyro.yaml
    direction_accgyro.yaml
    real_vs_complex.yaml
    synthetic_rotation.yaml

  src/
    datasets/
      hapt.py
      synthetic_rotation.py
    models/
      cnn1d.py
      tcn.py
      transformer.py
      real_ssm.py
      complex_ssm.py
      mamba3_model.py
    metrics/
      classification.py
      transition.py
      phase_analysis.py

  experiments/
    run_sensor_ablation.sh
    run_direction.sh
    run_real_vs_complex.sh
    run_synthetic_rotation.sh

  results/
    sensor_ablation/
    direction_classification/
    real_vs_complex/
    synthetic_rotation/
    phase_analysis/

  docs/
    result.md
    experiment_plan.md
```
