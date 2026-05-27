# Claude 작업 지시 프롬프트: Mamba-3 Internal State Phase 추출 및 GyroPhase Head 실험

## 목표

현재 목표는 **Mamba-3 기반 IMU 동작 전이 탐지 모델**에서 internal SSM state의 phase를 추출하고, 이를 **GyroPhase Head**에 활용할 수 있는지 실험하는 것이다.

핵심 연구 가설은 다음과 같다.

> Transformer IMU Calibrator에서 Transformer가 IMU short history를 인코딩해 동적 회전 상태를 추정한 것처럼, 본 연구는 Mamba-3/Complex SSM을 IMU 동작 전이의 회전성 상태 encoder로 사용한다. 그리고 기존 pooling head가 놓치는 hidden phase, gyro magnitude, rotation diversity 정보를 GyroPhase Head로 명시적으로 활용해 전이 방향 분류 성능을 개선한다.

---

## 1. 현재 문제

현재 Mamba-3는 fused kernel을 사용하기 때문에 forward hook만으로는 timestep별 internal SSM state, 즉 `h_t` 또는 complex state의 real/imag 값을 직접 얻기 어렵다.

fused kernel 내부에서 scan 중간 state가 계산되고 바로 사라질 가능성이 높기 때문에, PyTorch의 `register_forward_hook()`으로는 보통 다음 정도만 잡힌다.

```text
module input
module output
block output
projection output 일부
```

하지만 우리가 원하는 것은 다음이다.

```text
timestep별 internal SSM state h_t
complex state의 real/imag
hidden phase
hidden phase change
```

즉, 아래 값을 얻어야 한다.

```python
phase_t = atan2(Im(h_t), Re(h_t))
delta_phase_t = wrap(phase_t - phase_{t-1})
```

따라서 forward hook만으로 해결하려고 하지 말고, fused kernel을 바로 수정하지도 말고, 먼저 **분석용 unfused debug/reference path**를 구현한다.

---

## 2. 핵심 지시

절대 처음부터 fused Triton/CUDA kernel을 직접 수정하지 마라.  
그것은 마지막 수단이다.

fused kernel에서 internal `h`를 반환하게 만들면 forward output뿐 아니라 state tensor를 global memory에 저장해야 하고, 만약 그 state를 loss에 사용하면 backward까지 수정해야 할 수 있다. 지금 목적은 논문 실험을 위한 state 분석과 head 검증이므로, 느린 unfused reference path가 우선이다.

---

## 3. 구현 방향

Mamba-3 block 또는 wrapper에 다음 옵션을 추가한다.

```python
return_states: bool = False
use_fused: bool = True
debug_unfused: bool = False
```

동작 방식은 다음처럼 구성한다.

1. 일반 학습/추론에서는 기존 fused path를 그대로 사용한다.
2. `return_states=True` 또는 `debug_unfused=True`일 때는 fused kernel을 사용하지 않고, unfused reference scan을 실행한다.
3. unfused reference scan은 기존 Mamba-3 block의 projection 결과와 동일한 parameter를 사용해야 한다.
4. unfused path는 최종 output뿐 아니라 timestep별 internal state를 함께 반환해야 한다.
5. complex-valued state가 real/imag로 표현된다면 `h_real_seq`, `h_imag_seq`를 반환한다.
6. state shape는 가능하면 `[batch, time, state_dim]` 또는 `[batch, time, heads, state_dim]`처럼 time dimension이 명확하게 드러나게 한다.

예상 interface는 다음처럼 만들면 좋다.

```python
y = block(x)

y, state_dict = block(
    x,
    return_states=True,
    use_fused=False,
    debug_unfused=True,
)
```

`state_dict`에는 최소한 다음 값이 들어가야 한다.

```python
state_dict = {
    "h_real": h_real_seq,
    "h_imag": h_imag_seq,
    "phase": phase_seq,
    "delta_phase": delta_phase_seq,
}
```

---

## 4. Phase 계산

hidden state가 real/imag로 분리되어 있다면 phase는 다음처럼 계산한다.

```python
phase = torch.atan2(h_imag.float(), h_real.float())

delta = phase[:, 1:] - phase[:, :-1]
delta = torch.atan2(torch.sin(delta), torch.cos(delta))
delta_phase_abs = delta.abs()
```

gyro magnitude는 입력 IMU sequence에서 gyro channel을 사용해 계산한다.

```python
gyro = x[..., gyro_start:gyro_end]
gyro_mag = torch.sqrt((gyro ** 2).sum(dim=-1) + 1e-8)
```

window-level feature는 다음처럼 만든다.

```python
phase_score = delta_phase_abs.mean(dim=1)
gyro_score = gyro_mag[:, 1:].mean(dim=1)
```

그리고 Pearson correlation을 계산해서 hidden phase 변화량과 gyro magnitude가 얼마나 상관되는지 분석한다.

---

## 5. Fused path와 debug path 검증

unfused debug path가 실제 fused path와 같은 연산을 잘 재현하는지 확인해야 한다.

같은 input과 같은 checkpoint에 대해 다음을 비교하라.

```python
y_fused = block(x, use_fused=True)

y_debug, states = block(
    x,
    return_states=True,
    use_fused=False,
    debug_unfused=True,
)

max_abs_diff = (y_fused - y_debug).abs().max()
mean_abs_diff = (y_fused - y_debug).abs().mean()
cos_sim = cosine_similarity(y_fused.flatten(), y_debug.flatten())
```

완전히 같을 필요는 없다.  
bf16/fp32 차이를 고려했을 때 충분히 가까운지 확인하면 된다.

이 검증이 되어야 다음 주장을 할 수 있다.

> debug path에서 추출한 phase가 실제 Mamba-3 state와 같은 계열의 표현이다.

---

## 6. GyroPhase Head 실험 전략

처음부터 end-to-end로 하지 말고, 먼저 **offline/frozen encoder 방식**으로 진행한다.

권장 순서는 다음과 같다.

1. 기존 Mamba-3 + AvgPool 모델을 direction classification task로 학습한다.
2. 학습된 checkpoint를 로드한다.
3. unfused debug path로 train/val/test set의 hidden state를 추출한다.
4. hidden magnitude, hidden phase change, gyro magnitude, rotation diversity feature를 계산한다.
5. 기존 pooled representation `h_base`와 phase-aware feature `h_phase`를 concat한다.
6. encoder는 freeze하고 classifier head만 따로 학습한다.
7. 성능이 개선되는지 확인한다.

이 방식은 end-to-end는 아니지만 kernel backward 수정이 필요 없고, 논문 실험으로 빠르게 검증할 수 있다.

---

## 7. GyroPhase feature 설계

GyroPhase feature는 다음을 포함한다.

```python
m_t = abs(z_t)                       # hidden magnitude
p_t = abs(delta_phase_t)             # hidden phase change
q_t = gyro_magnitude_t               # gyro magnitude
d = rotation_diversity(window)        # window-level rotation diversity

interaction_1 = p_t * q_t
interaction_2 = p_t * d
interaction_3 = q_t * d
interaction_4 = p_t * q_t * d
```

time pooling은 mean, max, std를 모두 사용한다.

```python
h_phase = concat([
    mean(m_t), max(m_t), std(m_t),
    mean(p_t), max(p_t), std(p_t),
    mean(q_t), max(q_t), std(q_t),
    d,
    mean(p_t * q_t),
    mean(p_t * d),
    mean(q_t * d),
    mean(p_t * q_t * d),
])
```

최종 classifier 입력은 다음처럼 구성한다.

```python
h_final = concat([h_base, h_phase])
y_hat = classifier(h_final)
```

---

## 8. Rotation Diversity 구현

Rotation Diversity는 TIC 논문의 아이디어를 참고하여, window 안의 회전 정보가 충분히 다양한지 나타내는 feature로 사용한다.

처음에는 간단한 버전으로 구현해도 된다.

### 1차 구현: gyro std diversity

```python
RD = std(gyro_x) + std(gyro_y) + std(gyro_z)
```

### 2차 구현: gyro direction bin diversity

```text
1. gyro direction을 unit vector로 정규화
2. 방향 공간을 coarse bin으로 양자화
3. window 안에서 방문한 bin 개수를 count
4. count를 정규화해서 RD로 사용
```

1차 구현을 먼저 만들고, 효과가 보이면 2차 구현으로 확장한다.

---

## 9. 비교해야 할 모델

다음 모델을 비교한다.

```text
1. Transformer Encoder + AvgPool
2. Mamba-3 Encoder + AvgPool
3. Mamba-3 Encoder + Magnitude Head
4. Mamba-3 Encoder + Phase Head
5. Mamba-3 Encoder + GyroPhase Head
6. Mamba-3 Encoder + GyroPhase + Rotation Diversity Head
7. TCN baseline
```

가능하면 추가 비교도 수행한다.

```text
8. Transformer Encoder + GyroPhase Head
9. Complex-SSM + GyroPhase Head
10. Real-SSM + comparable head
```

---

## 10. 핵심 task

핵심 task는 **7-class transition direction classification**이다.

라벨은 다음과 같다.

```text
0: non-transition
1: stand-to-sit
2: sit-to-stand
3: sit-to-lie
4: lie-to-sit
5: stand-to-lie
6: lie-to-stand
```

Binary transition detection은 보조 실험으로 둔다.

---

## 11. 평가 지표

평가 지표는 다음을 반드시 포함한다.

```text
- Accuracy
- Macro F1
- Direction Macro F1: class 1~6 평균
- Worst-class F1
- Transition F1: binary로 변환했을 때 transition F1
- Opposite-pair confusion
- High-gyro subset F1
- High-RD subset F1
- Inference ms/window
- Params
```

Opposite-pair confusion은 다음 pair를 중점적으로 본다.

```text
stand-to-sit vs sit-to-stand
sit-to-lie vs lie-to-sit
stand-to-lie vs lie-to-stand
```

---

## 12. 성공 기준

### 최소 성공

```text
Mamba-3 + GyroPhase Head > Mamba-3 + AvgPool
Mamba-3 + GyroPhase Head > Transformer + AvgPool
```

### 좋은 성공

```text
Mamba-3 + GyroPhase + RD Head >= Mamba-3 + GyroPhase Head
High-gyro 또는 High-RD subset에서 TCN보다 우수
```

### 강한 성공

```text
Mamba-3 + GyroPhase Head가 전체 Direction Macro F1에서 TCN에 근접하거나 능가
```

---

## 13. 대안 플랜

만약 Mamba-3 internal state 추출이 끝내 어렵다면 대안도 준비한다.

### 대안 1. Mamba-3는 encoder baseline으로만 사용

Mamba-3는 다음 비교에만 사용한다.

```text
Transformer Encoder vs Mamba-3 Encoder
```

hidden phase 분석은 자체 Complex-SSM에서 수행한다.

논문 표현:

```text
Mamba-3 fused implementation에서는 intermediate state 접근이 제한적이므로,
복소수 state의 phase 분석은 동일한 complex update 원리를 갖는 lightweight Complex-SSM에서 수행하였다.
```

### 대안 2. Mamba-3-inspired Complex SSM Encoder를 제안 모델로 사용

공식 Mamba-3가 아니라 다음 구조를 제안 모델로 둔다.

```text
Mamba-3-inspired Complex SSM Encoder + GyroPhase Head
```

이 경우 공식 Mamba-3는 baseline 또는 관련 모델로 둔다.

### 대안 3. GyroPhase-TCN으로 전환

Mamba/GyroPhase 결과가 약하면 TCN backbone으로 전환한다.

```text
GyroPhase-TCN
```

TCN의 강한 baseline 성능을 활용하고, gyro magnitude, rotation diversity, phase-like temporal modulation을 붙여 모델 제안으로 전환한다.

---

## 14. 최종 산출물

최종 산출물은 다음이면 된다.

```text
1. Mamba-3 block/wrapper에 return_states/debug_unfused 옵션 추가
2. fused output vs debug output 비교 스크립트
3. hidden phase 추출 스크립트
4. gyro magnitude / rotation diversity feature 계산 스크립트
5. offline GyroPhase Head 학습 스크립트
6. 모델별 결과표
7. high-gyro / high-RD / opposite-pair confusion 분석표
```

---

## 15. 최종 목표

전체 목표는 “Mamba-3가 무조건 최고다”를 증명하는 것이 아니다.

최종 목표는 다음이다.

> Transformer IMU Calibrator에서 Transformer가 IMU short history를 인코딩해 동적 회전 상태를 추정한 것처럼, 본 연구는 Mamba-3/Complex SSM을 IMU 동작 전이의 회전성 상태 encoder로 사용한다. 그리고 기존 pooling head가 놓치는 hidden phase, gyro magnitude, rotation diversity 정보를 GyroPhase Head로 명시적으로 활용해 전이 방향 분류 성능을 개선한다.



---
추가 목표: complex phase뿐 아니라 selective scanning이 동작 전이 탐지에 도움이 되는지도 분석해야 한다.

따라서 debug_unfused path에서 hidden real/imag state뿐 아니라, selective scan의 input-dependent update 관련 값도 함께 반환하라. 구현에서 접근 가능한 이름이 무엇이든 상관없다. 예를 들어 delta, gate, update coefficient, retention coefficient, B/C projection, input gate, scan step size 등 state update 강도를 나타낼 수 있는 값을 state_dict에 포함하라.

반환 예시는 다음과 같다.

state_dict = {
    "h_real": h_real_seq,
    "h_imag": h_imag_seq,
    "phase": phase_seq,
    "delta_phase": delta_phase_seq,
    "selective_score": selective_score_seq,
    "retention": retention_seq,
    "update_strength": update_strength_seq,
}

selective_score는 구현상 정확한 내부 변수 이름과 달라도 된다. 중요한 것은 timestep별로 state가 얼마나 유지되거나 갱신되는지를 나타내는 proxy를 만드는 것이다.

분석할 상관관계는 다음이다.

1. corr(|Δphase|, gyro_magnitude)
2. corr(selective_score, gyro_magnitude)
3. corr(selective_score, |Δphase|)
4. transition vs non-transition에서 selective_score 평균 차이
5. high-gyro / high-RD subset에서 selective_score가 커지는지

추가 ablation도 수행하라.

- Real-Static SSM: real state, non-selective fixed update
- Real-Selective SSM: real state, input-dependent selective update
- Complex-Static SSM: complex state, non-selective fixed phase/update
- Complex-Selective SSM: complex state, input-dependent selective update

이 2×2 ablation으로 complex state의 효과와 selective scanning의 효과를 분리해서 보고해야 한다.

최종 head도 GyroPhase Head에서 Selective GyroPhase Head로 확장하라.

f_t = [
    |z_t|,
    |Δphase_t|,
    gyro_mag_t,
    rotation_diversity,
    selective_score_t,
    |Δphase_t| * gyro_mag_t,
    |Δphase_t| * selective_score_t,
    gyro_mag_t * selective_score_t,
    |Δphase_t| * gyro_mag_t * selective_score_t
]

비교 모델은 다음을 포함한다.

1. Mamba-3 + AvgPool
2. Mamba-3 + GyroPhase Head
3. Mamba-3 + Selective GyroPhase Head
4. Real-Static SSM
5. Real-Selective SSM
6. Complex-Static SSM
7. Complex-Selective SSM
8. TCN baseline
9. Transformer baseline

성공 기준은 다음이다.

- Selective model이 Static model보다 direction macro F1이 높다.
- Complex-Selective SSM이 Complex-Static SSM보다 높다.
- Selective score가 transition 구간에서 non-transition보다 크다.
- Selective score가 gyro magnitude 또는 |Δphase|와 양의 상관을 보인다.
- Mamba-3 + Selective GyroPhase Head가 Mamba-3 + AvgPool보다 성능이 향상된다.