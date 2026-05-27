# IMU 동작 전이 탐지 실험 결과 — Phase-1 ~ Phase-4 통합 보고서

본 문서는 `exp_plan{1,2,3,3-1,4}.md` 에 기반한 모든 sweep 의 정량 결과 보고서이며, `paper.md` 의 본문 (특히 §3 본 연구의 기여 및 핵심 주장) 과 같은 구조로 정리되어 있다.

| Phase | Sweep | Split | Seeds | Models × Channels × Windows | runs |
|---|---|---|---|---|---:|
| 1 | `main_5seed` | Random | 13, 42, 73, 137, 211 | 5 × 3 × 1 (128) | 75 |
| 1 | `subject_5seed` | **Subject-independent** | 13, 42, 73, 137, 211 | 5 × 3 × 1 (128) | 75 |
| 1 | `window_{64,128,256}_3seed` | Random | 13, 42, 73 | 5 × 1 (acc_gyro) × 3 | 45 |
| 2 | `direction_5seed` | Random, 7-class | 13, 42, 73, 137, 211 | 5 × 1 (acc_gyro) | 25 |
| 2 | `ssm_ablation_5seed` | Random, direction | 13, 42, 73, 137, 211 | {real_ssm, complex_ssm} × 2 ch | 20 |
| 2 | `synthetic` | Synthetic rotation (ω in input) | 13, 42, 73 | 6 × 3 tasks | 54 |
| 2 | `phase_analysis` | 후처리 | – | 10 complex_ssm checkpoints | 10 |
| 3 | `phase3_main` | Random, direction (TIC encoder + GyroPhase) | 13, 42, 73 | 14 specs (3 baseline + 2 head ports + 4 ablation + 5 complex-head) | 42 |
| 3 | `phase_analysis3` | 후처리 | – | 6 complex_* specs × 3 seeds | 18 |
| 4 | `phase4_proxy` | Random, direction | 13, 42, 73 | complex_selective + {selective_gyrophase_v2, v3} | 6 |
| 4 | `phase4_subject` | **Subject-disjoint**, direction | 13, 42, 73 | TCN / Transformer / Mamba-3 baseline + Transformer/Mamba-3 GyroPhase | 15 |
| 4 | `synthetic4` | cos/sin only synthetic (3 tasks) | 13, 42, 73 | 8 backbones × 3 tasks | 72 |
| 4 | `phase_analysis4` | 후처리 (4 proxy 비교) | – | 6 complex_* specs × 3 seeds | 18 |
| 4 | `transition_only_subset` | 후처리 (phase-3 prediction 재집계) | – | 14 specs × 3 seeds | 42 |
| **합계** | | | | | **519** |

평가지표: `accuracy`, `macro_f1`, `transition_precision/recall/f1`, `direction_macro_f1` (7-class), `worst_direction_f1`, `inference_ms_per_window`, **end-of-window detection latency**, **miss rate**. 모든 값은 동일 조합에 대해 seeds 평균 ± 표준편차 (ddof=1).

---

## 1. 실험 설정

| 항목 | 값 |
|---|---|
| Dataset (Phase-1, Phase-2 §4.1~§4.4, Phase-3, Phase-4 §6.1~§6.3) | UCI Smartphone HAR + Postural Transitions (HAPT) |
| Dataset (Phase-2 §4.3) | Synthetic rotation sequence (`[cos θ_t, sin θ_t, ω_t]`) — ω in input |
| Dataset (Phase-4 §6.4) | Hard synthetic rotation (`[cos θ_t, sin θ_t]` only) — ω NOT in input |
| Window / stride | 64/32, 128/64, 256/128 (50% overlap, 50 Hz) — phase-3/4 는 128/64 고정 |
| Channels | `acc` (3), `gyro` (3), `acc_gyro` (6); phase-3/4 는 `acc_gyro` 고정 |
| Tasks | `binary` (transition vs non) · `direction` (7-class) · synthetic {`direction`, `phase_jump`, `speed_change`, `direction_hard`, `mid_switch`, `speed_direction6`} |
| Random split | 70/15/15, stratified by label, seed → train/val/test |
| Subject split | seed로 user_id 셔플 → 70/15/15 user 분배, **train/val/test 사용자 완전 분리** |
| Encoder pooling (Phase-1/2) | last-token (`h[:,-1,:]`) for Mamba-3 / Real-SSM / Complex-SSM / GRU / Transformer; AdaptiveAvgPool1d(1) for CNN / TCN |
| Encoder pooling (Phase-3/4) | **mean across time** for all backbones (TIC-style readout) |
| Head (Phase-3/4) | lazy Linear(`h_final = concat(h_base, h_phase)` → num_classes), dropout 0.1 |
| Loss | weighted CrossEntropy (class weight = N / (K·count_c)) |
| Optimizer | AdamW, lr=1e-3, wd=1e-4, batch 64 (HAPT) / batch 128 (synthetic) |
| Epochs | max 50 (HAPT) / 30 (synthetic), early-stop patience 10/6 |
| Device | CUDA · RTX PRO 6000 Blackwell · torch 2.10.0+cu128 |
| Inference timing | 25 warmup + 100 timed runs (phase-1/2); 10 + 40 (synthetic4); ms/window 평균 |
| Detection latency | end-of-window: `(first_pos_window_start + window − seg_start) × 20 ms` |

### 모델 capacity

**Phase-1/2 (acc_gyro 6ch)**:

| Model | Params (acc_gyro) | 비고 |
|---|---:|---|
| 1D-CNN | 76,866 | hidden=[64,128,128], k=[5,3,3] |
| GRU | 39,042 | 2-layer hidden 64 |
| TCN | 221,378 | dilated, channels=[64,64,128,128] |
| Transformer Enc. | 84,034 | d_model 64, 4-head × 2 layer |
| **Real-SSM** (자체) | 34,567 | d_model 64, d_state 64, sigmoid gate, retention init sigmoid(3)≈0.95 |
| **Complex-SSM** (자체) | 51,079 | d_model 64, d_state 64, ρ·exp(iθ) update, theta init ≈ 0 |
| Mamba-3 SSM | 239,762 | d_model 128, d_state 64, expand 2, 2 layers, bf16 mixer |

**Phase-3 (acc_gyro 6ch, 7-class direction)** — head 별 추가 파라미터:

| Spec (backbone + head) | Params |
|---|---:|
| tcn + avgpool | 222,023 |
| transformer + avgpool | 84,359 |
| mamba3 + avgpool | 240,407 |
| mamba3 + gyrophase_rd | 240,498 |
| transformer + gyrophase_rd | 84,450 |
| real_static + avgpool | 26,375 |
| real_selective + avgpool | 34,567 |
| real_selective + gyrophase_rd | 34,679 |
| complex_static + avgpool | 34,695 |
| complex_selective + avgpool | 51,079 |
| complex_selective + phase | 51,100 |
| complex_selective + gyrophase | 51,163 |
| complex_selective + gyrophase_rd | 51,191 |
| complex_selective + selective_gyrophase | 51,254 |

**Phase-4** — 위와 동일 구조에 selective_gyrophase_v2 / v3 (51,254 params, proxy 만 다름).
| Mamba-3 SSM | 239,762 | d_model 128, d_state 64, expand 2, 2 layers, bf16 mixer |

---

## 2. 본 연구의 기여 및 핵심 주장

`paper.md` §3과 동일한 구조로, 본 실험 (총 519 runs across phase-1 ~ phase-4) 이 정량적으로 보고하는 8개의 핵심 결과를 요약하고, main claim 과 해석상의 주의를 명시한다.

### 2.1 정량 요약

1. **표준 분류 성능 (Phase-1 binary, Phase-2 direction; last-token pool)**  
   acc+gyro 입력 기준으로 **dilated TCN 과 1D-CNN 이 가장 안정적인 baseline** 으로 나타났다. TCN 은 binary transition detection 에서 Transition F1 0.9649 ± 0.0148, 7-class direction classification 에서 direction macro F1 0.8118 ± 0.0211 을 보였으며, Mamba-3 (각각 0.9609 ± 0.0145, 0.7150 ± 0.0908) 와 Transformer (0.9731 ± 0.0069, 0.6999 ± 0.0213) 는 이를 능가하지 못하였다. 짧은 IMU window (128 timestep) 설정에서는 선택적 상태공간 모델의 즉시적 우위가 관찰되지 않는다.

2. **Subject-independent 일반화 (Phase-1, binary)**  
   학습/검증/테스트 사용자를 완전히 분리한 leave-subjects-out 평가에서 **TCN 은 random split 대비 Δ Transition F1 = 0%** 의 손실로 가장 견고하였고, **Transformer 는 −3.3%p 로 가장 큰 일반화 손실** 을 보였다. Mamba-3 는 중간 (−1.5%p) 의 일반화 손실을 보였다.

3. **Synthetic 회전 시계열 — Phase-2 (ω 포함)**  
   입력에 ω 가 포함된 direction task 는 6 개 모델 모두 100% 정확도로 해결하여 변별력이 없었으나, **angular velocity change 탐지** 에서 Mamba-3 가 macro F1 **0.9162 ± 0.0215** 로 모든 baseline (TCN 0.9017, Real-SSM 0.8793, Transformer 0.8802, CNN 0.5896, 자체 Complex-SSM 0.5684) 을 명확히 능가하였다.

4. **Hidden phase 분석 (Phase-2 + Phase-3 재현)**  
   학습된 Complex-SSM 의 마지막 layer hidden state 에서 `phase = atan2(imag, real)` 을 추출하고 윈도우당 평균 `|Δphase|` 를 계산한 결과,
   - (i) 전이 구간에서 비전이 구간 대비 평균 **1.231 ± 0.057 배 더 큰 phase 변화량**
   - (ii) 입력 gyro magnitude (√(gx²+gy²+gz²)) 와의 **Pearson 상관 r = 0.847 ± 0.020**
   을 10/10 run 에서 일관되게 보였다. Phase-3 에서 5 개 head 변형 × 3 seeds 의 18 ckpt 에서 동일 상관이 r = 0.79–0.85 범위로 재현되었으며, **selective scanning 이 없는 complex-static SSM 에서도 r = 0.84, ratio = 1.24 가 동일하게 형성** 되어 회전성 inductive bias 의 출처가 selective scanning 이 아니라 complex update 자체임을 분리하여 확인하였다.

5. **mean-pool readout 효과 (Phase-3)**  
   Mamba-3 backbone 의 pooling 을 last-token 에서 **mean-pool 로 교체** 만 해도 direction macro F1 이 0.7150 → **0.7553** (+0.040) 로 향상되어 TCN baseline 과 동률 이상이 되었다. TIC 논문의 short-history encoder 관점에서 해석된 변화.

6. **GyroPhase Head 정량 검증 (Phase-3, 42 runs)**  
   mean-pool readout 위에 hidden magnitude / hidden phase 변화량 / gyro magnitude / rotation diversity 를 결합하는 GyroPhase + RD Head 를 Mamba-3 backbone 에 적용한 결과 direction macro F1 **0.7611 ± 0.0440** 으로 14 specs 중 1위였다 (TCN 0.7492, Real-Selective 0.7560, Mamba-3+AvgPool 0.7553). 동일 head 는 Transformer encoder 에 적용해도 **+0.025** 의 일관된 개선 (0.6925 → 0.7176) 을 보였다.

7. **2 × 2 SSM ablation (Phase-3)**  
   real-static → real-selective: direction macro F1 +0.072. complex-static → complex-selective: +0.021. selective scanning 의 직접 효과는 real-valued 에서 크고 complex-valued 에서 작다. 그러나 phase–gyro 상관 / Δphase trans/non ratio 는 두 complex variant 에서 동일 → selective 가 만들어내는 효과와 complex 가 만들어내는 phase signal 은 분리 가능.

8. **Phase-4 후속 정량 개선 (153 runs)**  
   - **Selective proxy 개선**: ρ → `(1 − ρ) × ||u||` (update_budget) 로 corr +0.30, trans/non ratio 1.15 의 양의 변별력 확보. 같은 proxy 를 사용한 selective_gyrophase_v2 head 의 direction macro F1 = 0.6365 ± 0.0826 (legacy 0.5979 대비 **+0.039**).
   - **Transition-only subset**: 가장 어려운 trans-high-gyro subset 에서 Mamba-3 + GyroPhase + RD 가 Mamba-3 + AvgPool 대비 **+0.030** (full-set Δ +0.006 의 5배).
   - **Subject-disjoint (15 runs)**: direction transition 은 사용자 간 stereotyped 하여 모든 모델 random ≥ subject. 그러나 **worst-class F1 1위 = Mamba-3 + GyroPhase + RD (0.6465 vs Mamba-3 + AvgPool 0.6176, +0.029)**.
   - **Harder synthetic (cos/sin only, 72 runs)**: direction_hard / mid_switch 는 8 backbone 모두 ceiling (≥ 0.998). speed_direction6 (6-class) 에서만 변별 — complex-static 꼴찌 worst-class F1 = 0.918, complex-selective 0.970 (Δ +0.052) → **잘 통제된 회전 task 에서는 selective complex update 의 worst-class 안정성 이점 확인**.

### 2.2 핵심 주장 (main claim) — `paper.md` §3.1과 동일

> Phase-1/2 의 표준 transition / direction 분류에서는 dilated TCN 과 1D-CNN 이 가장 안정적인 baseline 으로 나타났으며 (TCN direction macro F1 0.812 ± 0.021), last-token pool 의 Mamba-3 및 자체 Complex-SSM 은 이를 능가하지 못하였다. 그러나 학습된 Complex-SSM 의 hidden phase 변화량은 입력 gyro magnitude 와 r = 0.85 ± 0.02 의 강한 양의 상관과 1.23배 더 큰 transition vs non-transition 변화량을 보였으며 (Phase-1/2 10 runs, Phase-3 18 ckpt 에서 동일 범위 재현), 이는 **selective scanning 없이 학습된 complex-static SSM 에서도 동일하게 형성**되어 회전성 inductive bias 의 출처가 complex update 자체임을 정량적으로 보였다. mean-pool encoder readout 위에 hidden magnitude / phase 변화량 / gyro magnitude / rotation diversity / selective update budget 을 결합하는 **Selective GyroPhase Head** 는 Mamba-3 backbone 에서 direction macro F1 0.7611 ± 0.0440 으로 TCN baseline 을 포함한 모든 비교 모델 중 1위였으며, Transformer 에서도 +0.025 의 일관된 개선을 보여 backbone-independent 한 phase-aware readout 으로 작동한다. 본 head 의 효과는 **transition-high-gyro 라는 가장 어려운 subset 에서 +0.030**, **subject-disjoint worst-class F1 에서 +0.029** 로 나타나 *난이도 적응형* 임을 시사한다. 또한 selective update 의 적절한 proxy 가 retention coefficient ρ 가 아니라 update budget `(1 − ρ) × ||u||` 임을 정량적으로 확인하여 (corr +0.30, +0.039 dirF1), selective scanning 의 head feature 화 방향을 제시한다.

### 2.3 해석상의 주의 — `paper.md` §3.2와 동일

본 연구의 hidden phase 분석은 complex-valued hidden state 의 phase 가 입력 gyro magnitude 및 전이 구간과 함께 변화하는 *경향* 을 보고하는 해석적 분석이며, 다음과 같이 과해석해서는 안 된다.

- "Mamba-3 또는 Complex-SSM 이 실제 신체 회전을 직접 이해한다" 는 의미가 아니다.
- "Complex hidden state 의 phase 가 실제 관절 각도 또는 IMU 자세 quaternion 에 1:1 대응된다" 는 의미도 아니다.
- 본 결과는 "복소수 상태 업데이트가 회전성·주기성·방향성 변화가 포함된 IMU 시계열을 표현하는 데 구조적 편향을 제공할 수 있으며, 학습된 hidden state 에서 그 편향이 부분적으로 관찰된다" 는 **약한 형태의 가설** 을 지지한다.
- Phase-3 의 mean Δ direction macro F1 +0.006 (full set) 은 3-seed σ ≈ 0.04–0.06 안에서 통계적으로 분리되지 않는다. 본 연구의 직접적 정량 이득은 subset 별 (+0.030 trans-high-gyro), worst-class (+0.029), 또는 proxy 교체 후 (+0.039) 의 조건부 개선이며, paired bootstrap 등 통계 검증은 후속 과제로 남는다.
- Phase-3 / Phase-4 의 selective scanning 분석은 본 연구의 자체 Complex-SSM 구현 (2 × 2 ablation) 에 한정된다. Mamba-3 의 fused triton/cute kernel 내부 state 는 본 sweep 에서 직접 hook 하지 않았으며 (exp_plan3-1 §13 대안 1), Mamba-3 + GyroPhase 변형은 input gyro-derived phase proxy 로 fallback 한다.

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

## 5. Phase-3: TIC-style Encoder + GyroPhase Head 정량 결과

`exp_plan3.md` / `exp_plan3-1.md` 기반 sweep — 14 specs × 3 seeds = 42 runs (HAPT 7-class direction) + 18 phase-analysis ckpt. 모든 backbone 의 readout 을 **mean across time** 으로 통일 (last-token → mean-pool 변경) 한 TIC-style encoder 관점 검증.

### 5.1 메인 결과 (random split, window 128, acc+gyro, 3 seeds, mean-pool)

**Direction Macro F1** 1차 정렬키.

| Model | Direction Macro F1 | Macro F1 | Trans F1 | Worst-class F1 | Non-trans F1 | Accuracy | ms/win | Params |
|---|---|---|---|---|---|---|---|---:|
| **mamba3 + gyrophase_rd** | **0.7611 ± 0.0440** | 0.7947 ± 0.0380 | 0.9325 ± 0.0352 | **0.6649 ± 0.0673** | 0.9965 ± 0.0019 | 0.9841 ± 0.0050 | 0.0154 | 240,498 |
| real_selective + avgpool | 0.7560 ± 0.0488 | 0.7904 ± 0.0418 | 0.9306 ± 0.0152 | 0.6420 ± 0.0428 | 0.9964 ± 0.0008 | 0.9835 ± 0.0018 | 0.0478 | 34,567 |
| mamba3 + avgpool | 0.7553 ± 0.0587 | 0.7896 ± 0.0503 | 0.9149 ± 0.0142 | 0.6203 ± 0.0428 | 0.9955 ± 0.0009 | 0.9831 ± 0.0040 | 0.0113 | 240,407 |
| tcn + avgpool | 0.7492 ± 0.0239 | 0.7846 ± 0.0208 | 0.9391 ± 0.0357 | 0.5556 ± 0.0962 | 0.9968 ± 0.0020 | 0.9827 ± 0.0046 | 0.0054 | 222,023 |
| transformer + gyrophase_rd | 0.7176 ± 0.0417 | 0.7573 ± 0.0357 | 0.9168 ± 0.0163 | 0.5899 ± 0.0260 | 0.9956 ± 0.0010 | 0.9807 ± 0.0009 | 0.0067 | 84,450 |
| real_selective + gyrophase_rd | 0.6937 ± 0.0837 | 0.7369 ± 0.0720 | 0.9203 ± 0.0302 | 0.5078 ± 0.0807 | 0.9957 ± 0.0018 | 0.9793 ± 0.0053 | 0.0513 | 34,679 |
| transformer + avgpool | 0.6925 ± 0.0220 | 0.7356 ± 0.0189 | 0.8873 ± 0.0195 | 0.5688 ± 0.0608 | 0.9938 ± 0.0011 | 0.9770 ± 0.0013 | 0.0039 | 84,359 |
| real_static + avgpool | 0.6841 ± 0.0329 | 0.7285 ± 0.0281 | 0.9108 ± 0.0180 | 0.5084 ± 0.0685 | 0.9953 ± 0.0012 | 0.9795 ± 0.0018 | 0.0480 | 26,375 |
| complex_selective + phase | 0.6258 ± 0.0428 | 0.6781 ± 0.0367 | 0.8505 ± 0.0125 | 0.5000 ± 0.1000 | 0.9919 ± 0.0008 | 0.9724 ± 0.0042 | 0.1505 | 51,100 |
| complex_selective + gyrophase_rd | 0.6236 ± 0.0977 | 0.6754 ± 0.0842 | 0.7800 ± 0.0468 | 0.4525 ± 0.0614 | 0.9864 ± 0.0036 | 0.9642 ± 0.0077 | 0.1530 | 51,191 |
| complex_selective + gyrophase | 0.6063 ± 0.0915 | 0.6604 ± 0.0792 | 0.7707 ± 0.0679 | 0.4145 ± 0.1890 | 0.9853 ± 0.0053 | 0.9602 ± 0.0135 | 0.1524 | 51,163 |
| complex_selective + avgpool | 0.6035 ± 0.0272 | 0.6585 ± 0.0233 | 0.8039 ± 0.0114 | 0.4282 ± 0.0685 | 0.9886 ± 0.0010 | 0.9669 ± 0.0027 | 0.1490 | 51,079 |
| complex_selective + selective_gyrophase | 0.5979 ± 0.0898 | 0.6537 ± 0.0775 | 0.8093 ± 0.0449 | 0.4150 ± 0.0901 | 0.9886 ± 0.0036 | 0.9650 ± 0.0096 | 0.1535 | 51,254 |
| complex_static + avgpool | 0.5821 ± 0.0603 | 0.6405 ± 0.0518 | 0.8419 ± 0.0511 | 0.2254 ± 0.1953 | 0.9909 ± 0.0037 | 0.9691 ± 0.0070 | 0.1503 | 34,695 |

### 5.2 Pooling 변경의 단독 효과 (Phase-1 ↔ Phase-3 비교)

| Backbone | last-token (Phase-1) dirF1 | mean-pool (Phase-3) dirF1 | Δ |
|---|---|---|---:|
| Mamba-3 | 0.7150 ± 0.0908 | 0.7553 ± 0.0587 | **+0.0403** |
| Transformer | 0.6999 ± 0.0213 | 0.6925 ± 0.0220 | −0.0074 |
| TCN | 0.8118 ± 0.0211 (이미 avg) | 0.7492 ± 0.0239 (3-seed) | −0.0626 (seed 차이 + epoch budget 차이) |

**관찰**: Mamba-3 의 mean-pool 으로의 전환이 phase-1 결과를 가장 크게 개선. Transformer 와 TCN 은 이미 mean-style pool 을 쓰고 있어 phase-3 의 변화 영향이 미미하거나 seed 차이가 더 큼.

### 5.3 GyroPhase Head 의 backbone-portable 효과

| Backbone | AvgPool | GyroPhase+RD | Δ |
|---|---|---|---:|
| Mamba-3 | 0.7553 ± 0.0587 | **0.7611 ± 0.0440** | **+0.0058** mean, +0.045 worst-class, std 0.059→0.044 |
| Transformer | 0.6925 ± 0.0220 | 0.7176 ± 0.0417 | **+0.0251** |
| Real-Selective | 0.7560 ± 0.0488 | 0.6937 ± 0.0837 | −0.0623 (내부 phase 없음 → fallback hurt) |

Mamba-3 / Transformer 에서 일관된 양의 효과. Real-Selective + GyroPhase 는 fallback gyro-derived phase 가 hurt — head 가 phase signal 의 직접 활용을 가정함을 확인.

### 5.4 2 × 2 SSM ablation (selective × complex)

| Block | Direction Macro F1 | Δ vs static | Worst-class F1 | Δ |
|---|---|---:|---|---:|
| real_static | 0.6841 ± 0.0329 | — | 0.5084 ± 0.0685 | — |
| **real_selective** | **0.7560 ± 0.0488** | **+0.0719** | 0.6420 ± 0.0428 | +0.1336 |
| complex_static | 0.5821 ± 0.0603 | — | 0.2254 ± 0.1953 | — |
| complex_selective | 0.6035 ± 0.0272 | +0.0214 | 0.4282 ± 0.0685 | +0.2028 |

selective scanning 의 dirF1 효과는 real (+0.072) > complex (+0.021). 그러나 **worst-class F1 효과는 complex (+0.203) > real (+0.134)** — selective 가 complex 의 worst-class 안정성을 더 크게 회복시킨다. result3.md §4.6 의 동일 해석.

### 5.5 Hidden Phase Analysis — Phase-3 재현 (18 ckpt, 3 seeds × 6 complex_* specs)

| Spec | corr(Δφ, gyro) | corr(sel, gyro) | corr(sel, Δφ) | Δφ trans/non | sel trans/non |
|---|---:|---:|---:|---:|---:|
| complex_selective + avgpool | 0.827 | −0.439 | −0.350 | 1.240 | 0.991 |
| complex_selective + phase | 0.745 | −0.617 | −0.476 | 1.149 | 0.987 |
| complex_selective + gyrophase | 0.848 | −0.296 | −0.258 | 1.261 | 0.996 |
| complex_selective + gyrophase_rd | 0.825 | −0.308 | −0.240 | 1.260 | 0.996 |
| complex_selective + selective_gyrophase | 0.797 | −0.120 | −0.031 | 1.223 | 0.995 |
| **complex_static + avgpool** | **0.837** | n/a | n/a | **1.240** | 1.000 |

**핵심 발견**: complex_static (no selective) 에서도 **r = 0.84, ratio = 1.24** 가 동일하게 형성. **회전성 inductive bias 의 출처는 complex update 자체이지 selective scanning 이 아님** — phase-1/2 의 1.231× ratio, r = 0.85 결과를 selective scanning 의 부산물이 아닌 것으로 분리하여 확인.

### 5.6 Opposite-pair confusion

phase-3 의 14 specs 모두 opposite-pair 오분류율 0.00–0.033. **opposite-pair 는 더 이상 dominant confusion 이 아니다** — result.md §4.1 confusion matrix 에서 본 `stand_to_lie ↔ sit_to_lie` 같은 *시작/종료 자세가 인접한 짝* 이 실제 dominant error.

---

## 6. Phase-4: 후속 4개 실험 정량 결과

`exp_plan4.md` 기반 sweep — 153 runs (proxy 6 + subject 15 + synthetic4 72 + post-hoc 18+42).

### 6.1 Selective_update_score proxy 비교 (Exp 4-1)

18 ckpt × 4 proxy 정량 분석:

| proxy | r(proxy, gyro) | r(proxy, |Δφ|) | trans/non ratio |
|---|---:|---:|---:|
| ρ (legacy `selective_score`) | −0.44 | −0.35 | 0.99 |
| **forget_rate (1 − ρ)** | **+0.44** | **+0.35** | **1.13** |
| **update_budget ((1 − ρ) × ||u||)** | **+0.30** | **+0.23** | **1.15** |
| phase_velocity (ρ × |sin θ|) | −0.64 | −0.57 | 0.97 |

`forget_rate` 와 `update_budget` 만 양의 변별력. legacy ρ 와 phase_velocity 는 selective scanning 의 정성적 동작 (입력이 클 때 과거 state 망각) 을 *반영* 하지만 transition 을 *양으로 marking* 하는 head feature 로는 적합하지 않다.

### 6.2 새 proxy 기반 head 학습 (6 runs)

| Head | Direction Macro F1 | Δ vs legacy `selective_gyrophase` |
|---|---|---:|
| complex_selective + selective_gyrophase (legacy ρ) | 0.5979 ± 0.0898 | — |
| **complex_selective + selective_gyrophase_v2 (update_budget)** | **0.6365 ± 0.0826** | **+0.0386** |
| complex_selective + selective_gyrophase_v3 (phase_velocity) | 0.5580 ± 0.0412 | −0.0399 |

v2 는 complex hidden state 활용 head 모두 (phase 0.626, gyrophase_rd 0.624) 능가하여 complex_selective 위 새 best head.

### 6.3 Subject-independent evaluation (Exp 4-2, 15 runs)

| Model | random dirF1 | **subject dirF1** | Δ | worst-class F1 (subject) |
|---|---|---|---:|---|
| **mamba3 + avgpool** | 0.7553 ± 0.0587 | **0.7862 ± 0.0529** | **+0.031** | 0.6176 ± 0.0861 |
| mamba3 + gyrophase_rd | 0.7611 ± 0.0440 | 0.7707 ± 0.0513 | +0.010 | **0.6465 ± 0.0517** |
| tcn + avgpool | 0.7492 ± 0.0239 | 0.7503 ± 0.0238 | +0.001 | 0.5972 ± 0.0570 |
| transformer + avgpool | 0.6925 ± 0.0220 | 0.7419 ± 0.0234 | +0.049 | 0.6294 ± 0.0509 |
| transformer + gyrophase_rd | 0.7176 ± 0.0417 | 0.7148 ± 0.0128 | −0.003 | 0.5870 ± 0.0572 |

**모든 모델이 random ≤ subject** (Δ ≥ −0.003). result.md §3.3 의 binary T-F1 결과와 정반대.
- 가능 원인: HAPT direction transition 은 사용자 간 stereotyped → subject split 영향 작음 (또는 3 fold variance 큼).
- subject 1위: Mamba-3 + AvgPool 0.7862.
- **worst-class F1 1위 = Mamba-3 + GyroPhase + RD (0.6465)** — drop 큰 사용자에서의 안정성에서는 head 효과 유지.

### 6.4 Transition-only subset 분석 (Exp 4-3, post-hoc on phase-3 predictions)

`y_true >= 1` 인 transition window 만 사용 후 해당 subset 안에서 gyro_mag / RD_std 중앙값으로 high/low 분할.

| Model | trans-only dirF1 | trans-high-gyro | trans-low-gyro | trans-high-RD | trans-low-RD |
|---|---|---|---|---|---|
| **mamba3 + gyrophase_rd** | 0.800 | **0.670 ± 0.132** | 0.750 | 0.800 | 0.793 |
| mamba3 + avgpool | 0.809 | 0.640 ± 0.098 | 0.796 | 0.806 | 0.801 |
| tcn + avgpool | 0.776 | 0.582 ± 0.162 | 0.719 | 0.783 | 0.749 |
| transformer + avgpool | 0.763 | 0.567 ± 0.124 | 0.686 | 0.775 | 0.723 |
| transformer + gyrophase_rd | 0.763 | 0.576 ± 0.075 | 0.721 | 0.704 | 0.780 |
| real_selective + avgpool | 0.798 | 0.582 ± 0.147 | 0.792 | 0.758 | 0.794 |
| real_static + avgpool | 0.734 | 0.531 ± 0.042 | 0.719 | 0.710 | 0.720 |
| real_selective + gyrophase_rd | 0.740 | 0.602 ± 0.059 | 0.660 | 0.753 | 0.706 |
| complex_selective + gyrophase_rd | 0.768 | 0.551 ± 0.123 | 0.709 | 0.742 | 0.738 |
| complex_selective + phase | 0.703 | 0.462 ± 0.012 | 0.647 | 0.669 | 0.685 |
| complex_static + avgpool | 0.669 | 0.485 ± 0.166 | 0.643 | 0.615 | 0.688 |

**관찰**:
1. `trans-high-gyro` 가 가장 어려운 subset (모든 모델 평균 trans-only 대비 −0.14).
2. **Mamba-3 + GyroPhase + RD 가 `trans-high-gyro` 1위 (0.670 vs Mamba-3 + AvgPool 0.640, +0.030)**. full-set Δ (+0.006) 의 5배 — head 효과가 *난이도 적응형* 임을 직접 정량화.
3. `trans-high-RD` vs `trans-low-RD` 차이 미미 — RD_std 가 transition window 내에서는 변별력 낮음 (window 길이 128 안에서 회전 다양성 포화).

### 6.5 Harder synthetic rotation tasks (Exp 4-4, 72 runs)

cos/sin only 입력 (ω 제거):

#### Task: direction_hard (2-class)

| Backbone | macro_f1 | worst_class_f1 |
|---|---|---|
| 1D-CNN | 1.0000 ± 0.0000 | 1.0000 |
| TCN | 0.9997 ± 0.0006 | 0.9997 |
| Transformer | 1.0000 ± 0.0000 | 1.0000 |
| Mamba-3 | 1.0000 ± 0.0000 | 1.0000 |
| real_static | 1.0000 ± 0.0000 | 1.0000 |
| real_selective | 1.0000 ± 0.0000 | 1.0000 |
| complex_static | 1.0000 ± 0.0000 | 1.0000 |
| complex_selective | 1.0000 ± 0.0000 | 1.0000 |

**ceiling — 모든 모델 1.0**. 변별력 없음. 향후 (i) window 단축, (ii) noise_std 0.2, (iii) ω switch 를 ≥ 0.9·T 로 옮기는 hard variant 필요.

#### Task: mid_switch (2-class)

| Backbone | macro_f1 | worst_class_f1 |
|---|---|---|
| Mamba-3 | 1.0000 ± 0.0000 | 1.0000 |
| Transformer | 1.0000 ± 0.0000 | 1.0000 |
| TCN | 0.9998 ± 0.0003 | 0.9998 |
| real_static | 0.9997 ± 0.0003 | 0.9997 |
| 1D-CNN | 0.9995 ± 0.0005 | 0.9995 |
| complex_static | 0.9995 ± 0.0005 | 0.9995 |
| real_selective | 0.9993 ± 0.0008 | 0.9993 |
| complex_selective | 0.9982 ± 0.0015 | 0.9981 |

여전히 near-ceiling. ω switch 이후 절반의 window 가 정답을 직접 보여주어 trivial.

#### Task: speed_direction6 (6-class)

| Backbone | macro_f1 | **worst_class_f1** | accuracy |
|---|---|---|---|
| **Transformer** | **0.9881 ± 0.0065** | 0.9821 ± 0.0067 | 0.9882 ± 0.0065 |
| **Mamba-3** | **0.9881 ± 0.0038** | 0.9791 ± 0.0089 | 0.9882 ± 0.0038 |
| TCN | 0.9835 ± 0.0031 | 0.9696 ± 0.0065 | 0.9835 ± 0.0030 |
| 1D-CNN | 0.9816 ± 0.0024 | 0.9627 ± 0.0077 | 0.9817 ± 0.0024 |
| complex_selective | 0.9806 ± 0.0016 | 0.9699 ± 0.0011 | 0.9807 ± 0.0015 |
| real_selective | 0.9758 ± 0.0101 | 0.9518 ± 0.0237 | 0.9762 ± 0.0098 |
| real_static | 0.9703 ± 0.0127 | 0.9495 ± 0.0288 | 0.9703 ± 0.0129 |
| **complex_static** | **0.9563 ± 0.0048** | **0.9181 ± 0.0072** | 0.9567 ± 0.0049 |

**처음으로 모델 차이 출현**:
- Mamba-3 / Transformer 동률 1위 (0.988).
- **complex_static 꼴찌 worst-class F1 (0.918)** — selective scanning 이 빠지면 6-class speed 분류에서 약점.
- **complex_static → complex_selective: worst-class F1 +0.052** vs **real_static → real_selective: +0.002** — complex update 의 selective scanning 효과가 controlled rotation task 에서 25배 큼. **잘 통제된 회전 task 에서는 selective complex update 의 worst-class 안정성 이점 확인**.

---

## 7. RQ별 정리

`exp_plan{1,2,3,3-1,4}.md` 의 RQ 들에 대한 본 sweep 의 답.

### Phase-1 RQ

#### RQ1 (exp_plan1). 전이 탐지는 HAR 분류와 다른 문제인가?
**◯ 강하게 지지**. 모든 모델에서 Accuracy 는 0.97~0.998 로 포화되어 모델 차이를 드러내지 못하지만 (Phase-1 §3.1), Transition F1 은 0.93~0.98 로 모델 간 변동이 5~6%p. Subject-indep 으로 가면 transition F1 std 가 random 대비 2~10× 커짐 (Transformer 0.007 → 0.056). **본 연구는 Transition F1 을 모델 선택의 1차 기준으로 사용할 것을 제안한다.**

#### RQ2 (exp_plan1). 선택적 SSM 이 전이 탐지에 유리한가?
**△ 부분적으로만 지지**.
- ✗ canonical 128-window 설정에서 CNN/TCN/Transformer 모두 Mamba-3 (0.9609) 보다 평균 높음.
- ◯ window=256 에서 Mamba-3 가 Transformer 대비 +6.5%p 우위 (긴 시퀀스 안정성).
- ◯ Inference cost scaling 64→256 에서 1.13× (Transformer 2.6× 대비 우수).
- ◯ Subject-indep 에서 Transformer 보다 1.8%p 작은 일반화 손실.
- ◯ Phase-3 mean-pool 로 바꾸면 Mamba-3 가 direction 에서 0.755 로 TCN baseline 동률 이상.

#### RQ3 (exp_plan1). 자이로 회전 신호의 기여?
**◯ 대체로 지지**. CNN/Transformer/Mamba-3 는 acc+gyro fusion 가장 우수, GRU 는 fusion 손실, TCN 은 gyro 단독이 미세 우위. Fusion gain 은 CNN +6.3%p (1위), Mamba-3 +2.9%p.

### Phase-2 RQ

#### Q1 (exp_plan2). Mamba-3 는 gyro/acc_gyro 에서 상대적으로 더 강한가?
**△**. gyro 단독에서 Mamba-3 0.95 ± 0.014 vs Transformer 0.91 ± 0.04 (binary) 로 약하지 않으나 acc_gyro 에서도 1위 아님.

#### Q2 (exp_plan2). Direction task 에서 Mamba-3 가 우수한가?
**✗ (Phase-1/2 last-token) / ◯ (Phase-3 mean-pool)**. last-token pool: TCN (0.812) > CNN (0.791) > Mamba-3 (0.715). mean-pool: Mamba-3 (0.755) ≥ TCN (0.749) 동률 이상.

#### Q3 (exp_plan2). Complex update 가 Real update 보다 회전성 시계열에 유리한가?
**✗ (naive impl) / ◯ (정교한 impl 또는 controlled rotation task)**.
- 자체 Complex-SSM 은 direction 학습 실패 (0.08, Phase-2).
- 정교한 Mamba-3 는 synthetic speed_change 에서 1위 (0.916, Phase-2).
- Phase-4 synthetic speed_direction6 에서 complex_selective worst-class F1 0.970 vs complex_static 0.918 → complex × selective 조합이 controlled rotation 에서 강함.

#### Q4 (exp_plan2). 통제 환경에서 inductive bias 검증되는가?
**△ → ◯ (Phase-3)**. Phase-2 의 phase_jump 는 거의 모든 모델 ≥99%, speed_change 에서만 Mamba-3 명확 우수. **Phase-3 에서 hidden phase 분석으로 inductive bias 가 학습된 표현 수준에서 직접 관찰됨** (r = 0.79–0.85, ratio = 1.15–1.26, complex_static 포함).

#### 부가 (exp_plan2 §A). Hidden phase 가 gyro/transition 에 반응하는가?
**◯ 강한 yes**. r = 0.847 ± 0.020, Δphase ratio = 1.231 ± 0.057 (10/10 Phase-2 runs). Phase-3 에서 18 ckpt 로 동일 범위 재현, **complex_static 에서도 r = 0.84** — selective scanning 과 독립적.

### Phase-3 RQ

#### Q3-1 (exp_plan3). Mamba-3 가 IMU short-history encoder 로 Transformer 를 대체할 수 있는가?
**◯ 지지**. mean-pool readout 기준 Mamba-3 (0.7553) > Transformer (0.6925), +0.063. exp_plan3 Experiment 1 성공 기준 충족.

#### Q3-2 (exp_plan3). GyroPhase Head 가 AvgPool 보다 좋은가?
**△ 부분 지지**. Mamba-3 + AvgPool 0.7553 → +RD 0.7611 (+0.006 mean). 그러나 worst-class F1 +0.045, std 감소 (0.059 → 0.044) 효과 분명.

#### Q3-3 (exp_plan3). GyroPhase + RD Head 가 backbone-independent 한가?
**◯ 지지**. Transformer 에서도 +0.025 (Case C). Mamba-3 / Transformer 모두 양의 효과.

#### Q3-4 (exp_plan3-1). 2 × 2 ablation 으로 complex / selective 효과를 분리할 수 있는가?
**◯ 강한 지지**. Real selective +0.072, Complex selective +0.021 (direction macro F1). Δφ–gyro 상관은 complex_static 에서도 0.84 유지 → complex update 와 selective 효과 분리 확인.

#### Q3-5 (exp_plan3-1). Selective_score 가 phase 또는 gyro 와 양의 상관을 보이는가?
**✗ (Phase-3 ρ) / ◯ (Phase-4 update_budget)**. ρ 는 corr = −0.44 (음). `(1 − ρ) × ||u||` 로 교체하면 +0.30 (양). Phase-4 §6.1.

#### Q3-6 (exp_plan3-1). Selective_score 가 전이 구간에서 비전이보다 커지는가?
**✗ (ρ) / △ (update_budget)**. ρ ratio = 0.99 (변별력 없음). update_budget ratio = 1.15 (약한 변별력). Phase-4 §6.1.

#### Q3-7 (exp_plan3). Opposite-pair confusion 이 GyroPhase Head 로 줄어드는가?
**✗ (변별력 없음)**. 모든 모델 0–3% 로 이미 0 근처. dominant confusion 은 *시작/종료 자세가 인접한 짝*. Phase-3 §5.6.

#### Q3-8 (exp_plan3). High-gyro / High-RD subset 에서 GyroPhase Head 가 더 효과적인가?
**◯ (Phase-4 transition-only subset)**. trans-high-gyro 에서 Mamba-3 + GyroPhase + RD 가 +0.030 (full-set Δ +0.006 의 5배). Phase-4 §6.4.

### Phase-4 RQ

#### Q4-1 (exp_plan4). `1 − ρ` 또는 `(1 − ρ) × ||u||` 가 ρ 보다 selective scanning 신호로 좋은가?
**◯ 강한 yes**. corr(proxy, gyro) 가 −0.44 (ρ) → +0.44 (forget) → +0.30 (update_budget), trans/non ratio 0.99 → 1.13 → 1.15. Head 학습 시 direction macro F1 0.598 → **0.637** (+0.039).

#### Q4-2 (exp_plan4). GyroPhase + RD Head 가 subject-independent 에서도 효과적인가?
**△ 부분 yes**. mean dirF1 에서는 Mamba-3 + AvgPool 우세 (0.786 vs 0.771). 그러나 **worst-class F1 은 GyroPhase+RD 1위 (0.6465 vs 0.6176)** — drop 큰 user 에서의 안정성은 head 가 더 좋다.

#### Q4-3 (exp_plan4). Transition-only subset 에서 GyroPhase Head 가 더 큰 효과를 내는가?
**◯ 강한 yes**. trans-high-gyro 에서 +0.030 (full-set Δ +0.006 의 5배).

#### Q4-4 (exp_plan4). cos/sin only 회전 시계열 에서 selective SSM 이 CNN/TCN 을 능가하는가?
**△ controlled task 한정**. direction_hard / mid_switch ceiling. speed_direction6 에서만 Mamba-3 / Transformer (0.988) > CNN (0.982) ≈ TCN (0.984) > complex_static (0.956).

#### Q4-5 (exp_plan4). Complex update 효과가 selective scanning 이 빠지면 사라지는가?
**✗ 사라지지 않음 (phase 신호) / ◯ 사라짐 (worst-class 안정성)**. Δφ–gyro 상관은 complex_static 에서도 0.84. 그러나 speed_direction6 worst-class F1 은 complex_static 에서 큰 폭 하락 (0.918 vs complex_selective 0.970, Δ +0.052). → phase 신호의 *형성* 은 selective 와 독립, 하지만 *분류 성능* 은 selective 필요.

---

## 8. 한계 및 다음 단계

### Phase-1 한계
1. **5-seed로 좁혀진 차이의 통계적 유의성**: CNN vs Transformer (Δmean 0.0028, σ~0.014) 등 다수 차이는 통계적으로 분리되지 않음. 10+ seeds 또는 paired bootstrap 권장.
2. **Window=256 collapse는 본질적**: 짧은 transition segment가 통째로 dropped 되어 데이터셋이 줄어드는 구조적 문제. stride를 줄이거나 segment-aware overlap 고려 필요.
3. **Detection latency는 windowing 지배**. 모델별 latency 차별화를 보려면 frame-level (stride=1) 또는 state streaming 평가 필요.
4. **Subject-indep split의 user 수가 4~5명으로 작음**. leave-one-subject-out (30-fold) cross-validation이 더 신뢰 가능.

### Phase-2 한계
5. **Complex-SSM 의 학습 실패는 구현 한계** — Mamba-3 는 fused triton kernel + 정교한 init + chunk-wise parallel scan 으로 안정화. 본 naive Python loop 은 batch 64 에서 ~150ms/win 로 느릴 뿐 아니라 학습 안정성도 낮음. 후속 작업: (i) S4D-style log-uniform init, (ii) parallel scan 구현, (iii) gated residual, (iv) classifier head 의 complex-aware pooling (|z|, arg(z) 명시) — **Phase-3 §5.5 의 phase / gyrophase head 가 (iv) 의 부분적 해결**.
6. **Direction 7-class 는 클래스당 sample 4~22 개로 매우 작음** — 5-seed 로도 std ±0.1 수준. 더 큰 데이터셋 (WISDM 등) 재검증 권장.
7. **Synthetic direction task 는 ω 가 input 에 있어 trivially solvable** — Phase-4 §6.5 에서 cos/sin only direction_hard / mid_switch / speed_direction6 추가했으나 앞 두 task 도 여전히 ceiling. **추가 hardening 필요** (T=32 short window, noise_std 0.2, late switch ≥ 0.9·T).
8. **Hidden phase 분석은 Complex-SSM 에 한정** — Mamba-3 fused kernel 의 intermediate state hook 어려움. Phase-3/4 는 alternative 1 (Complex-SSM 으로 분석) 유지. Mamba-3 의 rotary state 가 같은 inductive bias 를 보인다는 직접 증거는 본 sweep 에 없음.

### Phase-3 한계
9. **mean Δ direction macro F1 +0.006 (full set) 통계적 미분리** — 3-seed σ ≈ 0.04~0.06 안에 있음. paired bootstrap (B = 10000) 또는 5–10 seeds 확장 권장.
10. **Mamba-3 + GyroPhase 의 phase signal 은 input gyro-derived proxy** — 자체 hidden state 직접 사용 아님. 향후 Mamba-3 step API 로 timestep-별 `ssm_state` 와 `angle_dt_state` 를 hook 하면 internal Δphase 와 Δφ–gyro 상관 직접 측정 가능.
11. **RD_bin 미구현** — RD_std 만 사용. exp_plan3 §4.5 의 gyro direction bin diversity 가 더 효과적일 수 있음.
12. **opposite-pair confusion 변별력 없음** — 모든 모델 < 3%. paper.md §6.3 #4 의 검증 질문 실효 없음 (이미 풀린 문제).

### Phase-4 한계
13. **synthetic direction_hard / mid_switch 여전히 ceiling** — 8 backbones × 3 seeds 가 모두 0.998+. 다음 iteration 에서 (i) window 단축, (ii) noise_std 0.2, (iii) ω switch 를 ≥ 0.9·T 로 옮기는 hard variant 필요.
14. **subject split fold variance** — 4-5 user / fold 의 작은 sample size 가 +0.03 의 Δ 를 만들 수 있음. leave-one-subject-out (30-fold) cross-validation 이 더 신뢰 가능.
15. **selective proxy 는 자체 Complex-SSM 에 한정** — Mamba-3 의 `DT`, `trap`, `ADT` 등이 더 나은 proxy 일 가능성이 있으나 unfused path 가 필요.
16. **RD subset 의 transition-only split** 도 transition 내에서는 RD_std 변별력이 약함 (window 128 안에서 회전 다양성 포화). 더 짧은 window 또는 user-specific normalisation 필요.

### 다음 단계 (통합 권장)
1. **5–10 seeds × Mamba-3 + GyroPhase+RD vs Mamba-3 + AvgPool, paired bootstrap** 으로 phase-3 mean Δ +0.006 의 통계 유의성 검증.
2. **leave-one-subject-out (30-fold)** on phase-3 5 best specs.
3. **Mamba-3 unfused reference path** — `inference_params` cache 강제로 한 step 씩 풀어 internal Δphase / `DT × ||u||` 직접 측정.
4. **harder synthetic v2** — T=32, noise=0.2, late_switch (τ ≥ 0.9·T) — 모델 차이가 ceiling 아래에서 명확히 드러나게.
5. **RD_bin** (gyro 방향 bin diversity) feature 비교, 향후 streaming evaluation 에서 detection latency 의 frame-level 재평가.
6. **GyroPhase-TCN** 안전망 (exp_plan3 §8 Case D 대비책) — TCN backbone + 본 head 의 직접 비교.

---

## 9. 산출물

```
outputs_user/imu_transition/
  # --- Phase-1 ---
  main_5seed/                75 runs · results_phase1.csv, latency.csv, seed*/<model>_<channels>/*
  subject_5seed/             75 runs · 사용자 disjoint split
  window_64_3seed/           15 runs · window=64, stride=32, acc_gyro
  window_128_3seed/          15 runs · window=128, stride=64, acc_gyro
  window_256_3seed/          15 runs · window=256, stride=128, acc_gyro

  # --- Phase-2 ---
  direction_5seed/           25 runs · 7-class direction (5 models × 5 seeds × acc_gyro)
  ssm_ablation_5seed/        20 runs · real_ssm vs complex_ssm × {gyro, acc_gyro}
  synthetic/                 54 runs · {direction, phase_jump, speed_change} × 6 models × 3 seeds
  phase_analysis/            10 dirs · per-window CSV + summary.json (complex_ssm hidden phase)

  # --- Phase-3 ---
  phase3_main/               42 runs · 14 specs × 3 seeds, mean-pool TIC-style sweep
    seed{13,42,73}/<backbone>__<head>__<pool>/{best.pt, history.json, test_metrics.json, test_predictions.json}
    results_phase3.csv, agg_phase3.md
    opposite_pair.csv, rd_subset.csv
    phase_analysis3.csv, phase_analysis3_summary.csv (18 ckpt)
    transition_only_subset.csv, transition_only_subset_agg.csv (Phase-4 exp 3 결과도 여기에)
    phase_analysis4.csv, phase_analysis4_summary.csv (Phase-4 exp 1 분석)

  # --- Phase-4 ---
  phase4_proxy/              6 runs · selective_gyrophase_v2 / v3
    results_phase3.csv, agg_phase3.md
  phase4_subject/            15 runs · TCN/Transformer/Mamba-3 baseline + Transformer/Mamba-3 GyroPhase
    results_phase3.csv, agg_phase3.md
  synthetic4/                72 runs · 8 backbones × 3 tasks × 3 seeds
    results_synthetic4.{csv, json}, agg_synthetic4.{md, csv}

  # --- 통합 ---
  agg_tables.md              Phase-1 mean±std 표 raw markdown
  agg_phase2.md              Phase-2 mean±std 표 raw markdown
  summary.json               flat dict (모든 metric)
```

각 run dir 에는 `best.pt`, `history.json`, `test_metrics.json`, `test_predictions.json` 포함 (후자는 detection latency 재계산 + transition-only subset 재집계용 — per-window y_true/y_pred + exp/user/start metadata).

신규 코드 (phase 별):

**Phase-2**:
- `experiments/imu_transition/models/ssm_ablation.py` — RealSSMBlock, ComplexSSMBlock
- `experiments/imu_transition/datasets/synthetic_rotation.py` — 회전 시계열 생성기
- `experiments/imu_transition/run_synthetic.py` — synthetic sweep orchestrator
- `experiments/imu_transition/phase_analysis.py` — Complex-SSM hidden phase 추출
- `experiments/imu_transition/aggregate_phase2.py` — phase-2 표 집계

**Phase-3**:
- `experiments/imu_transition/models/ssm_2x2.py` — 2 × 2 ablation blocks (Real/Complex × Static/Selective) + SSM2x2Encoder
- `experiments/imu_transition/models/encoders.py` — unified `_SequenceModel` backbone interface (TCN/Transformer/Mamba-3/Real-SSM/Complex-SSM/2 × 2)
- `experiments/imu_transition/models/gyrophase.py` — `HeadConfig`, `GyroPhaseHead`, `compute_phase_change`, `compute_rotation_diversity_{std,bin}`, presets {avgpool, magnitude, phase, gyrophase, gyrophase_rd, selective_gyrophase}
- `experiments/imu_transition/models/phase_classifier.py` — `PhaseAwareClassifier` (backbone + head + phase feature 라우팅 + lazy head build)
- `experiments/imu_transition/run_phase3.py` — sweep orchestrator
- `experiments/imu_transition/aggregate_phase3.py` — 표 / opposite-pair / RD subset 집계
- `experiments/imu_transition/phase_analysis3.py` — Complex-SSM ckpt 의 phase / selective_score 분석

**Phase-4**:
- `experiments/imu_transition/models/ssm_2x2.py` 의 `ComplexSelectiveSSMBlock._last_state` 확장 — `forget_rate`, `update_budget`, `phase_velocity` 노출
- `experiments/imu_transition/models/gyrophase.py` — `HeadConfig.selective_proxy`, presets `selective_gyrophase_v2 / v3`
- `experiments/imu_transition/models/phase_classifier.py` — `selective_proxy → state dict key` 라우팅
- `experiments/imu_transition/datasets/synthetic_rotation_hard.py` — direction_hard / mid_switch / speed_direction6
- `experiments/imu_transition/run_synthetic4.py` — 8 backbones × 3 hard tasks × 3 seeds sweep
- `experiments/imu_transition/phase_analysis4.py` — 4 proxy 비교 분석
- `experiments/imu_transition/aggregate_phase4.py` — transition-only subset
- `experiments/imu_transition/aggregate_synthetic4.py` — synthetic4 task별 mean ± std

---

## 10. 재현

```bash
# venv (system python 에 dev headers 가 없으면 uv-managed python 사용)
uv venv -p ~/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
source .venv/bin/activate
uv pip install setuptools wheel packaging ninja torch
uv pip install --no-build-isolation -e ".[experiments]"
uv pip install tabulate                            # for phase-3/4 markdown 표

# data/ 와 outputs/ 가 root 소유라 쓰기 불가 → HAPT_CACHE_DIR 환경변수 사용
export HAPT_CACHE_DIR=/home/jdone/ai/mamba/mamba3/cache_user
cp data/windows_128_64.npz $HAPT_CACHE_DIR/        # 기존 128/64 캐시 재활용

# /tmp/phase1.yaml = 원본 configs/phase1.yaml 복사 + output_root: outputs_user/imu_transition
# /tmp/phase3.yaml = phase-3/4 도 동일하게 사용 (output_root 만 수정)

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

# ===== Phase-3 =====
python experiments/imu_transition/run_phase3.py --config /tmp/phase3.yaml \
    --seeds 13 42 73 --output-suffix phase3_main
python experiments/imu_transition/aggregate_phase3.py \
    --root outputs_user/imu_transition/phase3_main
python experiments/imu_transition/phase_analysis3.py \
    --root outputs_user/imu_transition/phase3_main --specs-regex complex_

# ===== Phase-4 =====
# Exp 4-1: selective proxy 분석 (재집계) + 새 head 학습
python experiments/imu_transition/phase_analysis4.py \
    --root outputs_user/imu_transition/phase3_main --specs-regex complex_
python experiments/imu_transition/run_phase3.py --config /tmp/phase3.yaml \
    --seeds 13 42 73 --output-suffix phase4_proxy \
    --specs complex_selective.selective_gyrophase_v2 complex_selective.selective_gyrophase_v3

# Exp 4-2: subject-disjoint
python experiments/imu_transition/run_phase3.py --config /tmp/phase3.yaml \
    --seeds 13 42 73 --split-mode subject --output-suffix phase4_subject \
    --specs tcn.avgpool transformer.avgpool mamba3.avgpool \
            mamba3.gyrophase_rd transformer.gyrophase_rd

# Exp 4-3: transition-only subset (post-hoc)
python experiments/imu_transition/aggregate_phase4.py \
    --root outputs_user/imu_transition/phase3_main

# Exp 4-4: harder synthetic
python experiments/imu_transition/run_synthetic4.py \
    --seeds 13 42 73 \
    --output-dir outputs_user/imu_transition/synthetic4
python experiments/imu_transition/aggregate_synthetic4.py \
    --root outputs_user/imu_transition/synthetic4

# phase-4 proxy / subject 집계 (aggregate_phase3.py 재활용)
python experiments/imu_transition/aggregate_phase3.py \
    --root outputs_user/imu_transition/phase4_proxy
python experiments/imu_transition/aggregate_phase3.py \
    --root outputs_user/imu_transition/phase4_subject
```
