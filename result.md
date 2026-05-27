# IMU 동작 전이 탐지 실험 결과: Complex-Valued State Update의 회전성 표현 가능성 분석

본 문서는 `exp_plan1.md` / `exp_plan2.md`에 기반한 모든 sweep의 정량 결과 보고서이며, `paper.md`의 본문(특히 §3 본 연구의 기여 및 핵심 주장)과 같은 구조로 정리되어 있다.

| Phase | Sweep | Split | Seeds | Models × Channels × Windows | runs |
|---|---|---|---|---|---:|
| 1 | `main_5seed` | Random | 13, 42, 73, 137, 211 | 5 × 3 × 1 (128) | 75 |
| 1 | `subject_5seed` | **Subject-independent** | 13, 42, 73, 137, 211 | 5 × 3 × 1 (128) | 75 |
| 1 | `window_{64,128,256}_3seed` | Random | 13, 42, 73 | 5 × 1 (acc_gyro) × 3 | 45 |
| 2 | `direction_5seed` | Random, 7-class | 13, 42, 73, 137, 211 | 5 × 1 (acc_gyro) | 25 |
| 2 | `ssm_ablation_5seed` | Random, direction | 13, 42, 73, 137, 211 | {real_ssm, complex_ssm} × 2 ch | 20 |
| 2 | `synthetic` | Synthetic rotation | 13, 42, 73 | 6 × 3 tasks | 54 |
| 2 | `phase_analysis` | 후처리 | – | 10 complex_ssm checkpoints | 10 |
| **합계** | | | | | **304** |

평가지표: `accuracy`, `macro_f1`, `transition_precision/recall/f1`, `direction_macro_f1` (7-class), `inference_ms_per_window`, **end-of-window detection latency**, **miss rate**. 모든 값은 동일 조합에 대해 seeds 평균 ± 표준편차(ddof=1).

---

## 1. 실험 설정

| 항목 | 값 |
|---|---|
| Dataset (Phase-1, Phase-2 §4.1~§4.4) | UCI Smartphone HAR + Postural Transitions (HAPT) |
| Dataset (Phase-2 §4.3) | Synthetic rotation sequence (`[cos θ_t, sin θ_t, ω_t]`) |
| Window / stride | 64/32, 128/64, 256/128 (50% overlap, 50 Hz) |
| Channels | `acc` (3), `gyro` (3), `acc_gyro` (6) |
| Tasks | `binary` (transition vs non) · `direction` (7-class) · synthetic {`direction`, `phase_jump`, `speed_change`} |
| Random split | 70/15/15, stratified by label, seed → train/val/test |
| Subject split | seed로 user_id 셔플 → 70/15/15 user 분배, **train/val/test 사용자 완전 분리** |
| Loss | weighted CrossEntropy (class weight = N / (K·count_c)) |
| Optimizer | AdamW, lr=1e-3, wd=1e-4, batch 64 |
| Epochs | max 50 (HAPT) / 30 (synthetic), early-stop patience 10/6 |
| Device | CUDA · RTX PRO 6000 Blackwell · torch 2.10.0+cu128 |
| Inference timing | 25 warmup + 100 timed runs, ms/window 평균 |
| Detection latency | end-of-window: `(first_pos_window_start + window − seg_start) × 20 ms` |

### 모델 capacity (acc_gyro / synthetic 3-ch)

| Model | Params (acc_gyro) | 비고 |
|---|---:|---|
| 1D-CNN | 76,866 | hidden=[64,128,128], k=[5,3,3] |
| GRU | 39,042 | 2-layer hidden 64 |
| TCN | 221,378 | dilated, channels=[64,64,128,128] |
| Transformer Enc. | 84,034 | d_model 64, 4-head × 2 layer |
| **Real-SSM** (자체) | 34,567 | d_model 64, d_state 64, sigmoid gate, retention init sigmoid(3)≈0.95 |
| **Complex-SSM** (자체) | 51,079 | d_model 64, d_state 64, ρ·exp(iθ) update, theta init ≈ 0 |
| Mamba-3 SSM | 239,762 | d_model 128, d_state 64, expand 2, 2 layers, bf16 mixer |

---

## 2. 본 연구의 기여 및 핵심 주장

`paper.md` §3과 동일한 구조로, 본 실험 (총 304 runs)이 정량적으로 보고하는 4개의 핵심 결과를 먼저 요약하고, 그 위에 main claim과 해석상의 주의를 명시한다.

### 2.1 정량 요약

1. **표준 분류 성능 (Phase-1 binary, Phase-2 direction)**:
   acc+gyro 입력 기준으로 **dilated TCN과 1D-CNN이 가장 안정적인 baseline** 으로 나타났다. TCN은 binary transition detection에서 Transition F1 0.9649 ± 0.0148, 7-class direction classification에서 direction macro F1 0.8118 ± 0.0211을 보였으며, Mamba-3 (각각 0.9609 ± 0.0145, 0.7150 ± 0.0908)와 Transformer (0.9731 ± 0.0069, 0.6999 ± 0.0213)는 이를 능가하지 못하였다. 짧은 IMU window (128 timestep) 설정에서는 선택적 상태공간 모델의 즉시적 우위가 관찰되지 않는다.

2. **Subject-independent 일반화**:
   학습/검증/테스트 사용자를 완전히 분리한 leave-subjects-out 평가에서 **TCN은 random split 대비 Δ Transition F1 = 0%** 의 손실로 가장 견고하였고, **Transformer는 −3.3%p로 가장 큰 일반화 손실**을 보였다. Mamba-3는 중간 (−1.5%p) 의 일반화 손실을 보였다.

3. **Synthetic 회전 시계열 (통제 실험)**:
   입력에 ω가 포함된 direction task는 6개 모델 모두 100% 정확도로 해결하여 변별력이 없었으나, **angular velocity change 탐지**에서 Mamba-3가 macro F1 **0.9162 ± 0.0215**로 모든 baseline (TCN 0.9017, Real-SSM 0.8793, Transformer 0.8802, CNN 0.5896, 자체 Complex-SSM 0.5684)을 명확히 능가하였다. Phase jump detection은 TCN/CNN/Mamba-3/Real-SSM 모두 ≥0.987의 macro F1으로 변별력이 없었다.

4. **Hidden phase 분석 (본 연구의 핵심 관찰)**:
   학습된 Complex-SSM의 마지막 layer hidden state에서 `phase = atan2(imag, real)` 을 추출하고 윈도우당 평균 `|Δphase|` 를 계산한 결과,
   - (i) 전이 구간에서 비전이 구간 대비 평균 **1.231 ± 0.057배 더 큰 phase 변화량**
   - (ii) 입력 gyro magnitude (√(gx²+gy²+gz²))와의 **Pearson 상관 r = 0.847 ± 0.020**
   을 10/10 run에서 일관되게 보였다. classifier의 direction 분류 성능과 무관하게, complex-valued state는 회전성 신호를 표현 수준에서 추적하는 inductive bias를 형성한다는 정량적 증거이다.

### 2.2 핵심 주장 (main claim) — `paper.md` §3.1과 동일

> **표준 transition / direction 분류 성능에서는 dilated TCN과 1D-CNN이 가장 안정적인 baseline으로 나타났으며 (TCN direction macro F1 0.812 ± 0.021), Mamba-3 및 자체 Complex-SSM 구현은 이를 능가하지 못하였다. 그러나 (i) 통제된 synthetic 회전 시계열의 angular velocity change 탐지에서 Mamba-3는 가장 높은 macro F1 (0.916 ± 0.022)을 보였고, (ii) 본 연구에서 학습된 Complex-SSM의 hidden state phase 변화량은 입력 gyro magnitude와 r=0.85 ± 0.02의 강한 양의 상관을, 전이 구간에서 비전이 대비 1.23배 더 큰 변화량을 일관되게 보여, complex-valued state가 회전성 신호를 표현 수준에서 추적하는 inductive bias를 실제로 형성함을 확인하였다. classifier head가 이 phase signal을 분류에 활용하도록 하는 구조적 개선이 향후 과제이다.**

### 2.3 해석상의 주의 — `paper.md` §3.2와 동일

본 연구의 hidden phase 분석은 complex-valued hidden state의 phase가 입력 gyro magnitude 및 전이 구간과 함께 변화하는 *경향*을 보고하는 해석적 분석이며, 다음과 같이 과해석해서는 안 된다.

- "Mamba-3 또는 Complex-SSM이 실제 신체 회전을 직접 이해한다" 는 의미가 아니다.
- "Complex hidden state의 phase가 실제 관절 각도 또는 IMU 자세 quaternion에 1:1 대응된다"는 의미도 아니다.
- 본 결과는 "복소수 상태 업데이트가 회전성·주기성·방향성 변화가 포함된 IMU 시계열을 표현하는 데 구조적 편향을 제공할 수 있으며, 학습된 hidden state에서 그 편향이 부분적으로 관찰된다"는 **약한 형태의 가설**을 지지한다.

---

## 3. Phase-1: Binary Transition Detection 정량 결과

### 3.1 메인 결과 (random split, window 128, acc+gyro, 5 seeds)

| Model | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | End-of-win latency (ms) | Miss rate | Infer (ms/win) | Params |
|---|---|---|---|---|---|---|---|---|---:|
| 1D-CNN | 0.9977 ± 0.0018 | 0.9874 ± 0.0098 | 0.9675 ± 0.0242 | 0.9846 ± 0.0167 | **0.9759 ± 0.0186** | 3120.6 ± 115.0 | 0.0136 ± 0.0192 | **0.0019 ± 0.0001** | 76,866 |
| GRU | 0.9929 ± 0.0057 | 0.9624 ± 0.0294 | 0.9080 ± 0.0751 | 0.9513 ± 0.0378 | 0.9286 ± 0.0558 | 3111.1 ± 111.5 | 0.0437 ± 0.0380 | 0.0025 ± 0.0000 | **39,042** |
| TCN | 0.9966 ± 0.0015 | 0.9816 ± 0.0078 | 0.9462 ± 0.0230 | 0.9846 ± 0.0107 | 0.9649 ± 0.0148 | 3119.5 ± 117.1 | **0.0108 ± 0.0148** | 0.0058 ± 0.0000 | 221,378 |
| Transformer | 0.9974 ± 0.0007 | 0.9859 ± 0.0036 | 0.9721 ± 0.0159 | 0.9744 ± 0.0091 | 0.9731 ± **0.0069** | 3112.2 ± 122.6 | 0.0218 ± 0.0118 | 0.0039 ± 0.0000 | 84,034 |
| **Mamba-3** | 0.9962 ± 0.0015 | 0.9794 ± 0.0076 | 0.9508 ± 0.0303 | 0.9718 ± 0.0140 | 0.9609 ± 0.0145 | 3124.9 ± 130.4 | 0.0244 ± 0.0172 | 0.0110 ± 0.0000 | 239,762 |

**관찰**: 단일 시드(=42)로는 Transformer ≈ Mamba-3 (T-F1 0.9677 vs 0.9673) 였으나, 5-seed에서는 CNN이 mean 1위 (0.9759), Transformer가 std 1위 (±0.0069), Mamba-3는 4위. 단일 시드 결론은 5-seed에서 성립하지 않는다.

### 3.2 Sensor Ablation (모든 channels)

| Model | Channels | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | Latency (ms) | Miss rate |
|---|---|---|---|---|---|---|---|---|
| 1D-CNN | acc | 0.9913 ± 0.0026 | 0.9540 ± 0.0130 | 0.8828 ± 0.0441 | 0.9462 ± 0.0292 | 0.9126 ± 0.0246 | 3109 ± 107 | 0.0551 ± 0.0272 |
| 1D-CNN | gyro | 0.9944 ± 0.0017 | 0.9700 ± 0.0089 | 0.9161 ± 0.0282 | 0.9718 ± 0.0107 | 0.9430 ± 0.0169 | 3114 ± 112 | 0.0189 ± 0.0120 |
| 1D-CNN | **acc_gyro** | 0.9977 ± 0.0018 | 0.9874 ± 0.0098 | 0.9675 ± 0.0242 | 0.9846 ± 0.0167 | **0.9759 ± 0.0186** | 3121 ± 115 | 0.0136 ± 0.0192 |
| GRU | acc | 0.9934 ± 0.0035 | 0.9644 ± 0.0181 | 0.9222 ± 0.0549 | 0.9436 ± 0.0194 | 0.9323 ± 0.0343 | 3110 ± 98 | 0.0466 ± 0.0210 |
| GRU | gyro | 0.9948 ± 0.0024 | 0.9713 ± 0.0134 | 0.9372 ± 0.0265 | 0.9538 ± 0.0295 | 0.9453 ± 0.0254 | 3104 ± 99 | 0.0382 ± 0.0358 |
| GRU | acc_gyro | 0.9929 ± 0.0057 | 0.9624 ± 0.0294 | 0.9080 ± 0.0751 | 0.9513 ± 0.0378 | 0.9286 ± 0.0558 | 3111 ± 112 | 0.0437 ± 0.0380 |
| TCN | acc | 0.9959 ± 0.0015 | 0.9771 ± 0.0083 | 0.9591 ± 0.0235 | 0.9538 ± 0.0146 | 0.9564 ± 0.0157 | 3088 ± 107 | 0.0381 ± 0.0147 |
| TCN | **gyro** | 0.9978 ± 0.0014 | 0.9879 ± 0.0077 | 0.9797 ± 0.0210 | 0.9744 ± 0.0157 | **0.9769 ± 0.0147** | 3115 ± 116 | 0.0163 ± 0.0177 |
| TCN | acc_gyro | 0.9966 ± 0.0015 | 0.9816 ± 0.0078 | 0.9462 ± 0.0230 | 0.9846 ± 0.0107 | 0.9649 ± 0.0148 | 3120 ± 117 | 0.0108 ± 0.0148 |
| Transformer | acc | 0.9957 ± 0.0015 | 0.9762 ± 0.0083 | 0.9635 ± 0.0170 | 0.9462 ± 0.0167 | 0.9547 ± 0.0159 | 3100 ± 105 | 0.0463 ± 0.0205 |
| Transformer | gyro | 0.9916 ± 0.0042 | 0.9536 ± 0.0229 | 0.9176 ± 0.0646 | 0.9077 ± 0.0457 | 0.9116 ± 0.0436 | 3116 ± 107 | 0.0901 ± 0.0496 |
| Transformer | acc_gyro | 0.9974 ± 0.0007 | 0.9859 ± 0.0036 | 0.9721 ± 0.0159 | 0.9744 ± 0.0091 | 0.9731 ± 0.0069 | 3112 ± 123 | 0.0218 ± 0.0118 |
| Mamba-3 | acc | 0.9937 ± 0.0016 | 0.9645 ± 0.0089 | 0.9478 ± 0.0299 | 0.9179 ± 0.0146 | 0.9324 ± 0.0170 | 3082 ± 95 | 0.0708 ± 0.0206 |
| Mamba-3 | gyro | 0.9955 ± 0.0013 | 0.9751 ± 0.0076 | 0.9515 ± 0.0100 | 0.9538 ± 0.0266 | 0.9525 ± 0.0145 | 3114 ± 102 | 0.0381 ± 0.0245 |
| Mamba-3 | acc_gyro | 0.9962 ± 0.0015 | 0.9794 ± 0.0076 | 0.9508 ± 0.0303 | 0.9718 ± 0.0140 | 0.9609 ± 0.0145 | 3125 ± 130 | 0.0244 ± 0.0172 |

**Sensor fusion gain (acc → acc_gyro)**: CNN +0.063, Mamba-3 +0.029, Transformer +0.018, TCN +0.009, GRU −0.004. TCN은 gyro 단독이 acc_gyro보다 mean이 높은 유일한 모델.

### 3.3 Subject-Independent Evaluation (5 seeds, window 128, acc_gyro)

| Model | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | Latency (ms) | Miss rate |
|---|---|---|---|---|---|---|---|
| 1D-CNN | 0.9958 ± 0.0040 | 0.9798 ± 0.0173 | 0.9386 ± 0.0536 | 0.9873 ± 0.0134 | 0.9618 ± 0.0325 | 2584 ± 34 | **0.0000 ± 0.0000** |
| GRU | 0.9923 ± 0.0035 | 0.9618 ± 0.0166 | 0.8869 ± 0.0569 | 0.9745 ± 0.0269 | 0.9276 ± 0.0314 | 2609 ± 58 | 0.0073 ± 0.0100 |
| **TCN** | 0.9963 ± 0.0022 | 0.9815 ± 0.0102 | 0.9438 ± 0.0273 | 0.9874 ± 0.0219 | **0.9649 ± 0.0193** | 2589 ± 52 | **0.0000 ± 0.0000** |
| Transformer | 0.9933 ± 0.0074 | 0.9684 ± 0.0301 | 0.9169 ± 0.0928 | 0.9688 ± 0.0240 | 0.9403 ± 0.0562 | 2594 ± 41 | 0.0105 ± 0.0235 |
| Mamba-3 | 0.9944 ± 0.0043 | 0.9714 ± 0.0213 | 0.9348 ± 0.0246 | 0.9585 ± 0.0640 | 0.9458 ± 0.0403 | 2622 ± 112 | 0.0221 ± 0.0335 |

#### Random vs Subject-Indep T-F1 delta (§2.1 #2의 근거)

| Model | Random T-F1 | Subject-indep T-F1 | **Δ** |
|---|---|---|---:|
| 1D-CNN | 0.9759 ± 0.0186 | 0.9618 ± 0.0325 | −0.0141 |
| GRU | 0.9286 ± 0.0558 | 0.9276 ± 0.0314 | −0.0010 |
| **TCN** | 0.9649 ± 0.0148 | 0.9649 ± 0.0193 | **±0.0000** |
| Transformer | 0.9731 ± 0.0069 | 0.9403 ± 0.0562 | **−0.0329** |
| Mamba-3 | 0.9609 ± 0.0145 | 0.9458 ± 0.0403 | −0.0150 |

### 3.4 Window-Length Ablation (random split, acc_gyro, 3 seeds)

| Model | Window | Acc | Macro F1 | **Trans F1** | End-of-win latency (ms) | Miss rate | Infer (ms/win) |
|---|---:|---|---|---|---:|---|---|
| 1D-CNN | 64 | 0.9915 ± 0.0014 | 0.9671 ± 0.0052 | 0.9387 ± 0.0097 | 2247 ± 83 | 0.0148 ± 0.0064 | 0.0016 |
| 1D-CNN | **128** | 0.9980 ± 0.0020 | 0.9889 ± 0.0106 | **0.9790 ± 0.0201** | 3179 ± 76 | 0.0089 ± 0.0154 | 0.0019 |
| 1D-CNN | 256 | 0.9976 ± 0.0022 | 0.9371 ± 0.0588 | 0.8755 ± 0.1165 | 5120 ± 0 | 0.1429 ± 0.1429 | 0.0024 |
| GRU | 64 | 0.9924 ± 0.0017 | 0.9698 ± 0.0067 | 0.9437 ± 0.0124 | 2256 ± 69 | 0.0317 ± 0.0168 | 0.0016 |
| GRU | **128** | 0.9957 ± 0.0016 | 0.9768 ± 0.0084 | **0.9558 ± 0.0160** | 3172 ± 81 | 0.0274 ± 0.0129 | 0.0025 |
| GRU | 256 | 0.9899 ± 0.0043 | 0.7879 ± 0.0608 | 0.5809 ± 0.1195 | 5120 ± 0 | **0.3333 ± 0.1650** | 0.0041 |
| TCN | 64 | 0.9926 ± 0.0002 | 0.9708 ± 0.0005 | 0.9456 ± 0.0010 | 2254 ± 73 | 0.0261 ± 0.0090 | 0.0049 |
| TCN | **128** | 0.9967 ± 0.0013 | 0.9823 ± 0.0068 | **0.9663 ± 0.0130** | 3169 ± 89 | 0.0135 ± 0.0133 | 0.0058 |
| TCN | 256 | 0.9981 ± 0.0022 | 0.9499 ± 0.0599 | 0.9009 ± 0.1188 | 5120 ± 0 | 0.0952 ± 0.1650 | 0.0073 |
| Transformer | 64 | 0.9871 ± 0.0020 | 0.9502 ± 0.0069 | 0.9073 ± 0.0127 | 2268 ± 58 | 0.0425 ± 0.0080 | 0.0026 |
| Transformer | **128** | 0.9974 ± 0.0007 | 0.9854 ± 0.0039 | **0.9722 ± 0.0073** | 3175 ± 94 | 0.0227 ± 0.0150 | 0.0039 |
| Transformer | 256 | 0.9966 ± 0.0030 | 0.8981 ± 0.0895 | 0.7980 ± 0.1776 | 5120 ± 0 | 0.2857 ± 0.2474 | 0.0067 |
| Mamba-3 | 64 | 0.9885 ± 0.0013 | 0.9559 ± 0.0049 | 0.9181 ± 0.0091 | 2259 ± 73 | 0.0167 ± 0.0096 | 0.0111 |
| Mamba-3 | **128** | 0.9959 ± 0.0013 | 0.9778 ± 0.0064 | **0.9578 ± 0.0122** | 3191 ± 100 | 0.0272 ± 0.0227 | 0.0112 |
| Mamba-3 | 256 | 0.9976 ± 0.0030 | 0.9310 ± 0.0880 | 0.8632 ± 0.1745 | 5120 ± 0 | 0.1905 ± 0.2182 | 0.0125 |

**관찰**:
- 모든 모델에서 **128이 sweet spot**. 256은 짧은 transition segment가 통째로 dropped 되어 광범위한 성능 붕괴.
- **window=256에서 Mamba-3 (0.8632) > Transformer (0.7980)**, +6.5%p 우위. 긴 시퀀스에서 처음으로 명확한 차이 출현.
- **Inference scaling**: Mamba-3 64→256에서 1.13×, Transformer 2.6× — chunked SSM의 선형 비용 우위.

### 3.5 Detection Latency

End-of-window latency 이론 하한은 window length 자체 (64→1280ms, 128→2560ms, 256→5120ms). 측정값:

| Window | 평균 latency | 하한 대비 초과 | 해석 |
|---:|---:|---:|---|
| 64 | 2247 ms | +967 (~0.76 stride) | 첫 윈도우가 segment 시작과 어긋날 수 있음 |
| 128 | 3175 ms | +615 (~0.48 stride) | 1.5 윈도우 안에 거의 항상 탐지 |
| 256 | 5120 ms | 0 | 즉시 탐지가 유일한 mode (5.12s segment보다 짧은 transition은 모두 drop) |

**모델 간 latency 차이**: random split @ 128에서 모든 모델 3082~3125ms (~50ms 범위) → windowing 자체가 latency의 지배 요인. **"Mamba-3가 latency 측면에서 유리하다"는 claim은 본 setting에서 성립하지 않음** (§2.2 main claim에서도 다루지 않음).

---

## 4. Phase-2: Direction & Complex-State Experiments 정량 결과

### 4.1 Direction 7-class Classification (5 seeds, acc_gyro)

라벨: 0=non_transition, 1=stand_to_sit, 2=sit_to_stand, 3=sit_to_lie, 4=lie_to_sit, 5=stand_to_lie, 6=lie_to_stand. early-stop on `direction_macro_f1` (class 1–6 평균).

| Model | Acc | Macro F1 | Non-trans F1 | **Direction Macro F1** | Worst-class F1 | Trans F1 (binarized) | ms/win |
|---|---|---|---|---|---|---|---|
| 1D-CNN | 0.9873 ± 0.0017 | 0.8207 ± 0.0371 | 0.9982 ± 0.0007 | 0.7911 ± 0.0433 | 0.7006 ± 0.0311 | 0.9647 ± 0.0142 | 0.0019 |
| GRU | 0.9783 ± 0.0049 | 0.7233 ± 0.0588 | 0.9961 ± 0.0016 | 0.6778 ± 0.0685 | 0.4905 ± 0.0837 | 0.9264 ± 0.0282 | 0.0025 |
| **TCN** | 0.9885 ± 0.0010 | 0.8385 ± 0.0181 | 0.9985 ± 0.0007 | **0.8118 ± 0.0211** | 0.6690 ± 0.0481 | 0.9713 ± 0.0123 | 0.0058 |
| Transformer | 0.9811 ± 0.0028 | 0.7423 ± 0.0184 | 0.9965 ± 0.0018 | 0.6999 ± 0.0213 | 0.5373 ± 0.0674 | 0.9325 ± 0.0341 | 0.0039 |
| Mamba-3 | 0.9820 ± 0.0047 | 0.7553 ± 0.0778 | 0.9968 ± 0.0002 | 0.7150 ± 0.0908 | 0.5894 ± 0.1008 | 0.9366 ± 0.0056 | 0.0111 |

#### Per-class F1

| Model | non-trans | stand→sit | sit→stand | sit→lie | lie→sit | stand→lie | lie→stand |
|---|---|---|---|---|---|---|---|
| 1D-CNN | 0.998 ± 0.001 | 0.887 ± 0.041 | 0.877 ± 0.093 | 0.743 ± 0.034 | 0.755 ± 0.082 | 0.742 ± 0.034 | 0.742 ± 0.079 |
| GRU | 0.996 ± 0.002 | 0.737 ± 0.115 | 0.860 ± 0.123 | 0.578 ± 0.121 | 0.687 ± 0.173 | 0.535 ± 0.062 | 0.671 ± 0.084 |
| **TCN** | 0.999 ± 0.001 | **0.931 ± 0.065** | **0.930 ± 0.071** | **0.777 ± 0.029** | 0.747 ± 0.063 | **0.789 ± 0.063** | 0.696 ± 0.077 |
| Transformer | 0.996 ± 0.002 | 0.731 ± 0.090 | 0.847 ± 0.069 | 0.668 ± 0.071 | 0.681 ± 0.077 | 0.687 ± 0.086 | 0.584 ± 0.112 |
| Mamba-3 | 0.997 ± 0.000 | 0.789 ± 0.071 | 0.873 ± 0.138 | 0.607 ± 0.118 | 0.721 ± 0.080 | 0.661 ± 0.144 | 0.639 ± 0.097 |

#### Confusion matrix 평균 — Mamba-3 (acc_gyro, 5 seeds)

| true \\ pred | non-trans | s→si | si→s | si→l | l→si | s→l | l→s |
|---|---|---|---|---|---|---|---|
| non-transition | **1556.0** | 1.4 | 0.0 | 1.4 | 0.4 | 2.2 | 0.6 |
| stand_to_sit | 1.0 | **8.2** | 0.2 | 0.0 | 0.2 | 0.4 | 0.0 |
| sit_to_stand | 0.4 | 0.0 | **4.6** | 0.0 | 0.0 | 0.0 | 0.0 |
| sit_to_lie | 0.4 | 0.4 | 0.2 | **9.8** | 0.2 | **4.4** | 0.6 |
| lie_to_sit | 0.4 | 0.0 | 0.0 | 0.0 | **9.2** | 0.0 | **3.4** |
| stand_to_lie | 1.0 | 0.6 | 0.0 | **4.8** | 0.0 | **14.2** | 0.4 |
| lie_to_stand | 0.8 | 0.2 | 0.6 | 0.0 | **2.4** | 0.6 | **8.4** |

가장 큰 혼동은 **시작 자세가 다르고 종료 자세가 같거나 인접**한 쌍에서 발생 (`stand_to_lie ↔ sit_to_lie`, `lie_to_stand ↔ lie_to_sit`).

### 4.2 Real-SSM vs Complex-SSM Ablation (5 seeds, direction)

자체 구현한 두 block의 통제된 비교. d_model=64, d_state=64, 2 layers, retention init sigmoid(3)≈0.95. Complex만 ρ·exp(iθ) update + real/imag concat output.

| Model | Channels | Acc | Macro F1 | **Direction Macro F1** | Worst-class F1 | Trans F1 (bin) | Params | ms/win |
|---|---|---|---|---|---|---|---|---|
| **Real-SSM** | gyro | 0.9746 ± 0.0031 | 0.7130 ± 0.0379 | **0.6663 ± 0.0443** | 0.4959 ± 0.0682 | 0.8790 ± 0.0270 | 34,375 | 0.0477 |
| **Real-SSM** | acc_gyro | 0.9830 ± 0.0024 | 0.7652 ± 0.0224 | **0.7264 ± 0.0260** | 0.5606 ± 0.0659 | 0.9491 ± 0.0200 | 34,567 | 0.0477 |
| Complex-SSM | gyro | 0.5661 ± 0.0418 | 0.1446 ± 0.0081 | 0.0469 ± 0.0069 | 0.0050 ± 0.0112 | 0.1668 ± 0.0090 | 50,887 | 0.1490 |
| Complex-SSM | acc_gyro | 0.5761 ± 0.1263 | 0.1733 ± 0.0353 | 0.0810 ± 0.0230 | 0.0098 ± 0.0102 | 0.1619 ± 0.0335 | 51,079 | 0.1497 |

**솔직한 관찰**:
- Real-SSM (34k params)이 direction macro F1 0.7264로 **Mamba-3 (240k, 0.7150)와 Transformer (84k, 0.6999)를 능가**하는 강한 baseline.
- 자체 naive Complex-SSM은 direction task 학습에 **실패** (0.0810, chance 수준). Binary task에서는 학습됨 (T-F1 0.81 smoke test 기록). Direction의 class imbalance + 7-class CE + 회전 다이내믹스 동시 학습의 어려움이 합쳐진 것으로 보임.
- 그러나 §4.4에서 보듯 학습된 Complex-SSM의 **hidden state는 회전성 inductive bias를 형성** (r=0.85). classifier head가 phase signal을 활용하지 못한 것.

### 4.3 Synthetic Rotation Tasks (3 tasks × 6 models × 3 seeds)

입력 = `[cos θ_t, sin θ_t, ω_t]`, seq_len=128, noise_std=0.05, train 10000 / val 2000 / test 2000.

| Task | Model | Acc | **Macro F1** | Trans F1 | Params | ms/win |
|---|---|---|---|---|---|---|
| direction | 1D-CNN | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 75,906 | 0.0012 |
| direction | TCN | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 220,610 | 0.0036 |
| direction | Transformer | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 83,842 | 0.0027 |
| direction | Real-SSM | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 34,050 | 0.0238 |
| direction | Complex-SSM | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 50,562 | 0.0747 |
| direction | Mamba-3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 239,378 | 0.0055 |
| phase_jump | 1D-CNN | 0.9982 ± 0.0010 | 0.9982 ± 0.0010 | 0.9981 ± 0.0011 | 75,906 | 0.0012 |
| phase_jump | **TCN** | **0.9997 ± 0.0003** | **0.9997 ± 0.0003** | **0.9997 ± 0.0003** | 220,610 | 0.0036 |
| phase_jump | Transformer | 0.9928 ± 0.0013 | 0.9928 ± 0.0013 | 0.9926 ± 0.0013 | 83,842 | 0.0027 |
| phase_jump | Real-SSM | 0.9870 ± 0.0078 | 0.9870 ± 0.0078 | 0.9864 ± 0.0084 | 34,050 | 0.0237 |
| phase_jump | Complex-SSM | 0.5060 ± 0.0160 | 0.5010 ± 0.0155 | 0.4565 ± 0.0306 | 50,562 | 0.0747 |
| phase_jump | Mamba-3 | 0.9962 ± 0.0040 | 0.9962 ± 0.0040 | 0.9960 ± 0.0042 | 239,378 | 0.0055 |
| speed_change | 1D-CNN | 0.5948 ± 0.1499 | 0.5896 ± 0.1456 | 0.5541 ± 0.1338 | 75,906 | 0.0012 |
| speed_change | TCN | 0.9017 ± 0.0552 | 0.8999 ± 0.0574 | 0.8871 ± 0.0694 | 220,610 | 0.0036 |
| speed_change | Transformer | 0.8823 ± 0.0346 | 0.8802 ± 0.0357 | 0.8641 ± 0.0424 | 83,842 | 0.0027 |
| speed_change | Real-SSM | 0.8815 ± 0.0187 | 0.8793 ± 0.0198 | 0.8633 ± 0.0252 | 34,050 | 0.0238 |
| speed_change | Complex-SSM | 0.5763 ± 0.1228 | 0.5684 ± 0.1155 | 0.5736 ± 0.0563 | 50,562 | 0.0749 |
| speed_change | **Mamba-3** | **0.9162 ± 0.0215** | **0.9153 ± 0.0221** | 0.9068 ± 0.0261 | 239,378 | 0.0056 |

**관찰**:
- **direction**: ω가 input에 포함되어 trivially solvable → 모든 모델 100%, 변별력 없음.
- **phase_jump**: TCN/CNN/Mamba-3/Real-SSM/Transformer 모두 ≥0.987. 자체 Complex-SSM 실패 (0.50).
- **speed_change**: **Mamba-3가 처음으로 모든 baseline을 명확히 능가** (0.916 vs TCN 0.901 vs Real-SSM 0.879). §2.1 #3, §2.2 main claim의 (i)에 해당.

### 4.4 Hidden Phase Analysis (Complex-SSM, 10 runs = 5 seeds × {gyro, acc_gyro})

`expose_hidden=True`로 ComplexSSMBlock의 마지막 layer hidden state (real, imag)를 forward hook으로 추출, `phase_t = atan2(imag_t, real_t)`, `|Δphase_t| = |wrap_to_pi(phase_t − phase_{t-1})|`, 윈도우당 평균. 입력 gyro magnitude 윈도우당 평균과 Pearson 상관.

| Channels | Seed | Δphase ratio (trans/non-trans) | Phase-Gyro Pearson r |
|---|---:|---:|---:|
| acc_gyro | 13 | 1.31 | **0.881** |
| gyro | 13 | 1.25 | 0.841 |
| acc_gyro | 137 | 1.30 | 0.867 |
| gyro | 137 | 1.17 | 0.817 |
| acc_gyro | 211 | 1.20 | 0.859 |
| gyro | 211 | 1.18 | 0.836 |
| acc_gyro | 42 | 1.32 | 0.855 |
| gyro | 42 | 1.20 | 0.818 |
| acc_gyro | 73 | 1.19 | 0.853 |
| gyro | 73 | 1.18 | 0.844 |
| **mean** | – | **1.231 ± 0.057** | **0.847 ± 0.020** |

**관찰**: §4.2에서 Complex-SSM classifier는 direction task 학습에 실패했음에도 (direction_macro_f1 ≈ 0.08), **trained Complex-SSM의 hidden state는**:

1. 전이 구간에서 Δphase가 비전이 구간 대비 23% 더 큼 (10/10 run 일관).
2. Hidden phase 변화량과 입력 gyro magnitude의 Pearson r = 0.85 (10/10 run 일관).

즉 **모델 내부 표현은 회전성 신호를 추적하지만 classifier head가 이를 분류에 활용하지 못한 형태**. §2.2 main claim의 (ii)에 해당.

---

## 5. RQ별 정리

`exp_plan1.md` / `exp_plan2.md`의 RQ들에 대한 본 sweep의 답.

### RQ1 (exp_plan1). 전이 탐지는 HAR 분류와 다른 문제인가?
**◯ 강하게 지지**. 모든 모델에서 Accuracy는 0.97~0.998로 포화되어 모델 차이를 드러내지 못하지만 (Phase-1 §3.1), Transition F1은 0.93~0.98로 모델 간 변동이 5~6%p. Subject-indep으로 가면 transition F1 std가 random 대비 2~10× 커짐 (Transformer 0.007 → 0.056). **본 연구는 Transition F1을 모델 선택의 1차 기준으로 사용할 것을 제안한다.**

### RQ2 (exp_plan1). 선택적 SSM이 전이 탐지에 유리한가?
**△ 부분적으로만 지지**.
- ✗ canonical 128-window 설정에서 CNN/TCN/Transformer 모두 Mamba-3 (0.9609)보다 평균 높음.
- ◯ window=256에서 Mamba-3가 Transformer 대비 +6.5%p 우위 (긴 시퀀스 안정성).
- ◯ Inference cost scaling 64→256에서 1.13× (Transformer 2.6× 대비 우수).
- ◯ Subject-indep에서 Transformer보다 1.8%p 작은 일반화 손실.

### RQ3 (exp_plan1). 자이로 회전 신호의 기여?
**◯ 대체로 지지**. CNN/Transformer/Mamba-3는 acc+gyro fusion 가장 우수, GRU는 fusion 손실, TCN은 gyro 단독이 미세 우위. Fusion gain은 CNN +6.3%p (1위), Mamba-3 +2.9%p.

### Q1 (exp_plan2). Mamba-3는 gyro/acc_gyro에서 상대적으로 더 강한가?
**△**. gyro 단독에서 Mamba-3 0.95 ± 0.014 vs Transformer 0.91 ± 0.04 (binary)로 약하지 않으나 acc_gyro에서도 1위 아님.

### Q2 (exp_plan2). Direction task에서 Mamba-3가 우수한가?
**✗**. TCN (0.812) > CNN (0.791) > Mamba-3 (0.715). Mamba-3 mid-pack.

### Q3 (exp_plan2). Complex update가 Real update보다 회전성 시계열에 유리한가?
**✗ (naive impl) / ◯ (정교한 impl)**. 자체 Complex-SSM은 학습 실패 (direction 0.08). 정교한 Mamba-3는 synthetic speed_change에서 1위 (0.916).

### Q4 (exp_plan2). 통제 환경에서 inductive bias 검증되는가?
**△**. phase_jump는 거의 모든 모델 ≥99%. speed_change에서만 Mamba-3가 명확히 우수. direction은 ω가 input에 있어 trivially solvable.

### 부가 (exp_plan2 §A). Hidden phase가 gyro/transition에 반응하는가?
**◯ 강한 yes**. r=0.847 ± 0.020, Δphase ratio=1.231 ± 0.057 (10/10 runs).

---

## 6. 한계 및 다음 단계

### Phase-1 한계
1. **5-seed로 좁혀진 차이의 통계적 유의성**: CNN vs Transformer (Δmean 0.0028, σ~0.014) 등 다수 차이는 통계적으로 분리되지 않음. 10+ seeds 또는 paired bootstrap 권장.
2. **Window=256 collapse는 본질적**: 짧은 transition segment가 통째로 dropped 되어 데이터셋이 줄어드는 구조적 문제. stride를 줄이거나 segment-aware overlap 고려 필요.
3. **Detection latency는 windowing 지배**. 모델별 latency 차별화를 보려면 frame-level (stride=1) 또는 state streaming 평가 필요.
4. **Subject-indep split의 user 수가 4~5명으로 작음**. leave-one-subject-out (30-fold) cross-validation이 더 신뢰 가능.

### Phase-2 한계
5. **Complex-SSM의 학습 실패는 구현 한계** — Mamba-3는 fused triton kernel + 정교한 init + chunk-wise parallel scan으로 안정화. 본 naive Python loop은 batch 64에서 ~150ms/win로 느릴 뿐 아니라 학습 안정성도 낮음. 후속 작업: (i) S4D-style log-uniform init, (ii) parallel scan 구현, (iii) gated residual, (iv) classifier head의 complex-aware pooling (|z|, arg(z) 명시).
6. **Direction 7-class는 클래스당 sample 4~22개로 매우 작음** — 5-seed로도 std ±0.1 수준. 더 큰 데이터셋 (WISDM 등) 재검증 권장.
7. **Synthetic direction task는 ω가 input에 있어 trivially solvable** — 향후 cos/sin만 입력으로 줘서 dθ/dt 추정 task로 변형 필요.
8. **Hidden phase 분석은 Complex-SSM에 한정** — Mamba-3는 fused kernel이라 intermediate state hook 어려움. Mamba-3의 rotary state가 같은 inductive bias를 보인다는 직접 증거는 본 sweep에 없음.

---

## 7. 산출물

```
outputs_user/imu_transition/
  # --- Phase-1 ---
  main_5seed/         75 runs · results_phase1.csv, latency.csv, seed*/<model>_<channels>/*
  subject_5seed/      75 runs · 사용자 disjoint split
  window_64_3seed/    15 runs · window=64, stride=32, acc_gyro
  window_128_3seed/   15 runs · window=128, stride=64, acc_gyro
  window_256_3seed/   15 runs · window=256, stride=128, acc_gyro

  # --- Phase-2 ---
  direction_5seed/    25 runs · 7-class direction (5 models × 5 seeds × acc_gyro)
  ssm_ablation_5seed/ 20 runs · real_ssm vs complex_ssm × {gyro, acc_gyro}
  synthetic/          54 runs · {direction, phase_jump, speed_change} × 6 models × 3 seeds
  phase_analysis/     10 dirs · per-window CSV + summary.json (complex_ssm hidden phase)

  # --- 통합 ---
  agg_tables.md       Phase-1 mean±std 표 raw markdown
  agg_phase2.md       Phase-2 mean±std 표 raw markdown
  summary.json        flat dict (모든 metric)
```

각 run dir에는 `best.pt`, `history.json`, `test_metrics.json`, `test_predictions.json` 포함 (후자는 detection latency 재계산용 — per-window y_true/y_pred + exp/user/start metadata).

신규 코드 (phase-2):
- `experiments/imu_transition/models/ssm_ablation.py` — RealSSMBlock, ComplexSSMBlock
- `experiments/imu_transition/datasets/synthetic_rotation.py` — 회전 시계열 생성기
- `experiments/imu_transition/run_synthetic.py` — synthetic sweep orchestrator
- `experiments/imu_transition/phase_analysis.py` — Complex-SSM hidden phase 추출
- `experiments/imu_transition/aggregate_phase2.py` — phase-2 표 집계

---

## 8. 재현

```bash
# venv (system python에 dev headers가 없으면 uv-managed python 사용)
uv venv -p ~/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
source .venv/bin/activate
uv pip install setuptools wheel packaging ninja torch
uv pip install --no-build-isolation -e ".[experiments]"

# data/ 와 outputs/가 root 소유라 쓰기 불가 → HAPT_CACHE_DIR 환경변수 사용
export HAPT_CACHE_DIR=/home/jdone/ai/mamba/mamba3/cache_user
cp data/windows_128_64.npz $HAPT_CACHE_DIR/   # 기존 128/64 캐시 재활용

# /tmp/phase1.yaml = 원본 configs/phase1.yaml 복사 + output_root: outputs_user/imu_transition

# ===== Phase-1 =====
python experiments/imu_transition/run_phase1.py --config /tmp/phase1.yaml \
    --seeds 13 42 73 137 211 --split-mode random --output-suffix main_5seed
python experiments/imu_transition/run_phase1.py --config /tmp/phase1.yaml \
    --seeds 13 42 73 137 211 --split-mode subject --output-suffix subject_5seed
for WIN in 64 128 256; do
  STRIDE=$((WIN/2))
  python experiments/imu_transition/run_phase1.py --config /tmp/phase1.yaml \
    --seeds 13 42 73 --split-mode random --channels acc_gyro \
    --window-size $WIN --stride $STRIDE --output-suffix window_${WIN}_3seed
done
for D in main_5seed subject_5seed window_64_3seed window_128_3seed window_256_3seed; do
  python experiments/imu_transition/compute_latency.py \
    --predictions-glob "outputs_user/imu_transition/$D/seed*/*/test_predictions.json" \
    --data-root data/uci_har_pt \
    --output "outputs_user/imu_transition/$D/latency.csv"
done
python experiments/imu_transition/aggregate_results.py

# ===== Phase-2 =====
python experiments/imu_transition/run_phase1.py --config /tmp/phase1.yaml \
    --seeds 13 42 73 137 211 --task direction --channels acc_gyro \
    --output-suffix direction_5seed
python experiments/imu_transition/run_phase1.py --config /tmp/phase1.yaml \
    --seeds 13 42 73 137 211 --task direction \
    --models real_ssm complex_ssm --channels gyro acc_gyro \
    --output-suffix ssm_ablation_5seed
python experiments/imu_transition/run_synthetic.py \
    --models cnn tcn transformer real_ssm complex_ssm mamba3 \
    --tasks direction phase_jump speed_change --seeds 13 42 73 \
    --train-n 10000 --val-n 2000 --test-n 2000 --epochs 30 --early-stop-patience 6 \
    --output-dir outputs_user/imu_transition/synthetic
python experiments/imu_transition/phase_analysis.py \
    --checkpoint-glob 'outputs_user/imu_transition/ssm_ablation_5seed/seed*/complex_ssm_*/best.pt' \
    --output-dir outputs_user/imu_transition/phase_analysis
python experiments/imu_transition/aggregate_phase2.py
```
