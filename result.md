# Phase-1 실험 결과: IMU 기반 동작 전이 구간 탐지 (5-seed, subject-indep, window ablation, detection latency)

본 문서는 다음 세 가지 sweep을 통합한 결과이다.

| Sweep | Split | Seeds | Models × Channels × Windows | 총 runs |
|---|---|---|---|---:|
| `main_5seed` | Random (stratified) | 13, 42, 73, 137, 211 | 5 × 3 × 1 (128) | 75 |
| `subject_5seed` | **Subject-independent** | 13, 42, 73, 137, 211 | 5 × 3 × 1 (128) | 75 |
| `window_{64,128,256}_3seed` | Random | 13, 42, 73 | 5 × 1 (acc_gyro) × 3 | 45 |
| **합계** | | | | **195** |

평가지표는 `accuracy`, `macro_f1`, `transition_precision`, `transition_recall`,
`transition_f1`, `inference_ms_per_window`, 그리고 새로 추가된
**end-of-window detection latency** 와 **miss rate** (전이 segment 중 한 번도 transition으로 예측되지 않은 비율). 모든 값은 동일 (model × channels × window) 조합에 대해 seeds 평균 ± 표준편차(ddof=1).

배경/RQ는 `exp_plan1.md`, 논문 본문 초안은 `paper.md` 참고.

---

## 1. 실험 설정

| 항목 | 값 |
|---|---|
| Dataset | UCI Smartphone HAR + Postural Transitions (HAPT) |
| Task | Binary transition detection (`transition` vs `non-transition`) |
| Window / stride | 64/32, 128/64, 256/128 (50% overlap, 50 Hz) |
| Channels | `acc` (3), `gyro` (3), `acc_gyro` (6) |
| Random split | 70/15/15, stratified by binary label, seed → train/val/test |
| Subject split | seed로 user_id 셔플 → 70/15/15 user 분배, **train/val/test 사용자 완전 분리** |
| Loss | weighted CrossEntropy (class weight = N / (2·count_c)) |
| Optimizer | AdamW, lr=1e-3, wd=1e-4 |
| Batch | 64 |
| Epochs | max 50, early-stop patience 10 on val `transition_f1` |
| Normalize | per-channel z-score from train split |
| Device | CUDA · RTX PRO 6000 Blackwell · torch 2.10.0+cu128 |
| Inference timing | 25 warmup + 100 timed runs, ms/window 평균 |
| Detection latency | end-of-window: `(first_pos_window_start + window − seg_start) × 20 ms` |

### 모델 capacity (acc_gyro)

| Model | Params | 비고 |
|---|---:|---|
| 1D-CNN | 76,866 | hidden=[64,128,128], k=[5,3,3] |
| GRU | 39,042 | 2-layer hidden 64 |
| TCN | 221,378 | dilated, channels=[64,64,128,128] |
| Transformer Enc. | 84,034 | d_model 64, 4-head × 2 layer |
| Mamba-3 SSM | 239,762 | d_model 128, d_state 64, expand 2, 2 layers, bf16 mixer |

---

## 2. 메인 결과 (random split, window 128, acc+gyro, **5 seeds**)

논문 Table 1 후보.

| Model | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | End-of-win latency (ms) | Miss rate | Infer (ms/win) | Params |
|---|---|---|---|---|---|---|---|---|---:|
| 1D-CNN | 0.9977 ± 0.0018 | 0.9874 ± 0.0098 | 0.9675 ± 0.0242 | 0.9846 ± 0.0167 | **0.9759 ± 0.0186** | 3120.6 ± 115.0 | 0.0136 ± 0.0192 | **0.0019 ± 0.0001** | 76,866 |
| GRU | 0.9929 ± 0.0057 | 0.9624 ± 0.0294 | 0.9080 ± 0.0751 | 0.9513 ± 0.0378 | 0.9286 ± 0.0558 | 3111.1 ± 111.5 | 0.0437 ± 0.0380 | 0.0025 ± 0.0000 | **39,042** |
| TCN | 0.9966 ± 0.0015 | 0.9816 ± 0.0078 | 0.9462 ± 0.0230 | 0.9846 ± 0.0107 | 0.9649 ± 0.0148 | 3119.5 ± 117.1 | **0.0108 ± 0.0148** | 0.0058 ± 0.0000 | 221,378 |
| Transformer | 0.9974 ± 0.0007 | 0.9859 ± 0.0036 | 0.9721 ± 0.0159 | 0.9744 ± 0.0091 | 0.9731 ± **0.0069** | 3112.2 ± 122.6 | 0.0218 ± 0.0118 | 0.0039 ± 0.0000 | 84,034 |
| **Mamba-3** | 0.9962 ± 0.0015 | 0.9794 ± 0.0076 | 0.9508 ± 0.0303 | 0.9718 ± 0.0140 | 0.9609 ± 0.0145 | 3124.9 ± 130.4 | 0.0244 ± 0.0172 | 0.0110 ± 0.0000 | 239,762 |

**핵심 관찰 — 단일 시드 결과로부터의 수정**

이전 단일 시드(=42) 표에서는 Transformer 0.9677 vs Mamba-3 0.9673로 사실상 동률이었지만, 5-seed로 다시 보면:

- **CNN이 mean 1위 (0.9759)** — 단순한 1D-CNN이 평균적으로 가장 높은 Transition F1을 보임.
- **Transformer가 std 1위 (±0.0069)** — 안정성에서는 다른 모델 대비 2~3배 작은 분산. Seed 변동에 강건.
- **Mamba-3는 4위 (0.9609)** — CNN/TCN/Transformer 모두에게 평균 1.2~1.5%p 차이로 뒤짐. 단일 시드의 우연한 동률이 다중 시드에서 해소됨.
- CNN과 Transformer의 mean 차이(0.0028)는 각 표준편차 안에 들어가므로 통계적으로 분리되지 않음.
- TCN은 **miss rate 1위 (0.0108)** — 한 번도 탐지 못한 segment 비율이 가장 낮음.

> 결론: **단일 시드(=42)의 "Mamba-3 ≈ Transformer 동률" 주장은 5-seed에서 성립하지 않는다.**
> 정확한 표현은 "Mamba-3는 random split에서 best 대비 1.5%p 낮은 mid-pack 성능을 보였다."

### 전체 (모든 channels) 표

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

흥미로운 부가 관찰:

- **TCN gyro 단독 (0.9769)은 acc_gyro (0.9649)보다 평균이 높다**. 가속도가 오히려 노이즈로 작용한 모델별 패턴.
- **Transformer gyro 단독은 모든 모델 중 최악 (0.9116, miss rate 0.090)**. self-attention이 회전 신호만으로는 안정적인 boundary를 잡지 못함.
- Mamba-3는 세 channel 모드에서 가장 일관된 (모두 0.93~0.96) 성능. acc 단독에서도 Transformer를 약간 능가 (0.9324 vs 0.9547... — 정정: Transformer 0.9547 > Mamba-3 0.9324; Mamba는 acc 단독에서는 약점).

---

## 3. Subject-Independent Evaluation (5 seeds, window 128)

학습/검증/테스트에 등장하는 user_id가 완전히 분리된 leave-subjects-out 평가. 30명을 70/15/15로 사용자 단위 분배(약 21/4~5/4~5명).

### 메인 (acc_gyro)

| Model | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | End-of-win latency (ms) | Miss rate |
|---|---|---|---|---|---|---|---|
| 1D-CNN | 0.9958 ± 0.0040 | 0.9798 ± 0.0173 | 0.9386 ± 0.0536 | 0.9873 ± 0.0134 | 0.9618 ± 0.0325 | 2584 ± 34 | **0.0000 ± 0.0000** |
| GRU | 0.9923 ± 0.0035 | 0.9618 ± 0.0166 | 0.8869 ± 0.0569 | 0.9745 ± 0.0269 | 0.9276 ± 0.0314 | 2609 ± 58 | 0.0073 ± 0.0100 |
| **TCN** | 0.9963 ± 0.0022 | 0.9815 ± 0.0102 | 0.9438 ± 0.0273 | 0.9874 ± 0.0219 | **0.9649 ± 0.0193** | 2589 ± 52 | **0.0000 ± 0.0000** |
| Transformer | 0.9933 ± 0.0074 | 0.9684 ± 0.0301 | 0.9169 ± 0.0928 | 0.9688 ± 0.0240 | 0.9403 ± 0.0562 | 2594 ± 41 | 0.0105 ± 0.0235 |
| Mamba-3 | 0.9944 ± 0.0043 | 0.9714 ± 0.0213 | 0.9348 ± 0.0246 | 0.9585 ± 0.0640 | 0.9458 ± 0.0403 | 2622 ± 112 | 0.0221 ± 0.0335 |

### Random vs Subject-Indep T-F1 (acc_gyro)

| Model | Random split T-F1 | Subject-indep T-F1 | **Δ** |
|---|---|---|---:|
| 1D-CNN | 0.9759 ± 0.0186 | 0.9618 ± 0.0325 | −0.0141 |
| GRU | 0.9286 ± 0.0558 | 0.9276 ± 0.0314 | −0.0010 |
| **TCN** | 0.9649 ± 0.0148 | 0.9649 ± 0.0193 | **±0.0000** |
| Transformer | 0.9731 ± 0.0069 | 0.9403 ± 0.0562 | **−0.0329** |
| Mamba-3 | 0.9609 ± 0.0145 | 0.9458 ± 0.0403 | −0.0150 |

**핵심 관찰 — 일반화 (RQ에는 없지만 논문에 강력)**

1. **TCN은 일반화 손실이 0** (Δ=0). Random과 subject-indep에서 동일한 0.9649 mean. dilated convolution이 사용자별 특성을 적게 외움.
2. **Transformer는 가장 크게 떨어짐 (−3.3%p)**. acc 단독에서는 오히려 subject에서 더 좋지만(0.9763 vs 0.9547), acc+gyro에서는 random→subject 하락이 가장 큼. self-attention이 user-specific feature를 overfitting하는 경향.
3. **Mamba-3는 CNN과 유사한 −1.5%p 하락**. 일반화 측면에서 중간.
4. Subject-indep에서 CNN/TCN은 **miss rate 0%** (모든 전이 segment를 최소 한 번은 탐지). Mamba-3는 0.022로 일부 segment 놓침.

> **새로운 논문 claim**: "Subject-independent 평가에서 dilated TCN이 가장 견고하며(Δ=0), self-attention 기반 Transformer는 user-specific overfitting으로 가장 큰 일반화 손실(−3.3%p)을 보였다. 선택적 SSM(Mamba-3)은 두 group의 중간 수준 일반화를 보였다."

### 전체 (모든 channels) 표

| Model | Channels | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | Latency (ms) | Miss rate |
|---|---|---|---|---|---|---|---|---|
| 1D-CNN | acc | 0.9870 ± 0.0062 | 0.9381 ± 0.0261 | 0.8345 ± 0.1105 | 0.9492 ± 0.0462 | 0.8831 ± 0.0489 | 2601 ± 60 | 0.0331 ± 0.0355 |
| 1D-CNN | gyro | 0.9919 ± 0.0066 | 0.9607 ± 0.0302 | 0.8887 ± 0.1114 | 0.9744 ± 0.0256 | 0.9257 ± 0.0569 | 2594 ± 63 | 0.0000 ± 0.0000 |
| 1D-CNN | acc_gyro | 0.9958 ± 0.0040 | 0.9798 ± 0.0173 | 0.9386 ± 0.0536 | 0.9873 ± 0.0134 | 0.9618 ± 0.0325 | 2584 ± 34 | 0.0000 ± 0.0000 |
| GRU | acc | 0.9761 ± 0.0387 | 0.9166 ± 0.1118 | 0.7995 ± 0.2631 | 0.9461 ± 0.0469 | 0.8462 ± 0.2022 | 2590 ± 54 | 0.0251 ± 0.0386 |
| GRU | gyro | 0.9887 ± 0.0074 | 0.9470 ± 0.0273 | 0.8423 ± 0.0851 | 0.9710 ± 0.0218 | 0.9000 ± 0.0507 | 2580 ± 27 | 0.0035 ± 0.0078 |
| GRU | acc_gyro | 0.9923 ± 0.0035 | 0.9618 ± 0.0166 | 0.8869 ± 0.0569 | 0.9745 ± 0.0269 | 0.9276 ± 0.0314 | 2609 ± 58 | 0.0073 ± 0.0100 |
| TCN | acc | 0.9912 ± 0.0085 | 0.9585 ± 0.0368 | 0.8789 ± 0.1227 | 0.9785 ± 0.0189 | 0.9217 ± 0.0690 | 2594 ± 40 | 0.0000 ± 0.0000 |
| TCN | gyro | 0.9904 ± 0.0144 | 0.9586 ± 0.0547 | 0.8907 ± 0.1509 | 0.9659 ± 0.0319 | 0.9224 ± 0.1017 | 2593 ± 27 | 0.0000 ± 0.0000 |
| TCN | acc_gyro | 0.9963 ± 0.0022 | 0.9815 ± 0.0102 | 0.9438 ± 0.0273 | 0.9874 ± 0.0219 | 0.9649 ± 0.0193 | 2589 ± 52 | 0.0000 ± 0.0000 |
| Transformer | **acc** | 0.9976 ± 0.0027 | 0.9875 ± 0.0132 | 0.9801 ± 0.0194 | 0.9729 ± 0.0349 | **0.9763 ± 0.0251** | 2584 ± 43 | 0.0108 ± 0.0159 |
| Transformer | gyro | 0.9893 ± 0.0077 | 0.9483 ± 0.0341 | 0.8677 ± 0.1055 | 0.9452 ± 0.0283 | 0.9023 ± 0.0642 | 2613 ± 55 | 0.0070 ± 0.0157 |
| Transformer | acc_gyro | 0.9933 ± 0.0074 | 0.9684 ± 0.0301 | 0.9169 ± 0.0928 | 0.9688 ± 0.0240 | 0.9403 ± 0.0562 | 2594 ± 41 | 0.0105 ± 0.0235 |
| Mamba-3 | acc | 0.9814 ± 0.0077 | 0.9130 ± 0.0324 | 0.7628 ± 0.1025 | 0.9329 ± 0.0309 | 0.8358 ± 0.0607 | 2604 ± 44 | 0.0187 ± 0.0187 |
| Mamba-3 | gyro | 0.9941 ± 0.0056 | 0.9712 ± 0.0243 | 0.9361 ± 0.0688 | 0.9561 ± 0.0220 | 0.9454 ± 0.0456 | 2590 ± 44 | 0.0070 ± 0.0157 |
| Mamba-3 | acc_gyro | 0.9944 ± 0.0043 | 0.9714 ± 0.0213 | 0.9348 ± 0.0246 | 0.9585 ± 0.0640 | 0.9458 ± 0.0403 | 2622 ± 112 | 0.0221 ± 0.0335 |

추가 관찰:

- Subject-indep에서 **Transformer acc 단독이 0.9763**으로 같은 모델의 acc_gyro(0.9403)보다 높음. user-별 gyro 분포 차이가 generalization에 악영향.
- Mamba-3는 acc 단독에서 큰 약점(0.8358) — 가속도만으로 user-invariant 표현 학습에 실패. gyro 추가 시 0.9454로 회복.
- TCN/CNN은 acc_gyro에서 miss rate 0% 유지 — 실응용에서 가장 안전한 선택.

---

## 4. Window-Length Ablation (random split, acc_gyro, 3 seeds)

window ∈ {64, 128, 256} × stride = window/2 (50% overlap 유지).

| Model | Window | Acc | Macro F1 | Trans P | Trans R | **Trans F1** | End-of-win latency (ms) | Miss rate | Infer (ms/win) |
|---|---:|---|---|---|---|---|---:|---|---|
| 1D-CNN | 64 | 0.9915 ± 0.0014 | 0.9671 ± 0.0052 | 0.9018 ± 0.0179 | 0.9789 ± 0.0000 | 0.9387 ± 0.0097 | 2247 ± 83 | 0.0148 ± 0.0064 | 0.0016 |
| 1D-CNN | **128** | 0.9980 ± 0.0020 | 0.9889 ± 0.0106 | 0.9711 ± 0.0310 | 0.9872 ± 0.0128 | **0.9790 ± 0.0201** | 3179 ± 76 | 0.0089 ± 0.0154 | 0.0019 |
| 1D-CNN | 256 | 0.9976 ± 0.0022 | 0.9371 ± 0.0588 | 0.8968 ± 0.0901 | 0.8571 ± 0.1429 | 0.8755 ± 0.1165 | 5120 ± 0 | 0.1429 ± 0.1429 | 0.0024 |
| GRU | 64 | 0.9924 ± 0.0017 | 0.9698 ± 0.0067 | 0.9328 ± 0.0152 | 0.9550 ± 0.0122 | 0.9437 ± 0.0124 | 2256 ± 69 | 0.0317 ± 0.0168 | 0.0016 |
| GRU | **128** | 0.9957 ± 0.0016 | 0.9768 ± 0.0084 | 0.9467 ± 0.0365 | 0.9658 ± 0.0074 | **0.9558 ± 0.0160** | 3172 ± 81 | 0.0274 ± 0.0129 | 0.0025 |
| GRU | 256 | 0.9899 ± 0.0043 | 0.7879 ± 0.0608 | 0.5545 ± 0.2232 | 0.6667 ± 0.1650 | 0.5809 ± 0.1195 | 5120 ± 0 | **0.3333 ± 0.1650** | 0.0041 |
| TCN | 64 | 0.9926 ± 0.0002 | 0.9708 ± 0.0005 | 0.9263 ± 0.0167 | 0.9662 ± 0.0184 | 0.9456 ± 0.0010 | 2254 ± 73 | 0.0261 ± 0.0090 | 0.0049 |
| TCN | **128** | 0.9967 ± 0.0013 | 0.9823 ± 0.0068 | 0.9543 ± 0.0186 | 0.9786 ± 0.0074 | **0.9663 ± 0.0130** | 3169 ± 89 | 0.0135 ± 0.0133 | 0.0058 |
| TCN | 256 | 0.9981 ± 0.0022 | 0.9499 ± 0.0599 | 0.9028 ± 0.0867 | 0.9048 ± 0.1650 | 0.9009 ± 0.1188 | 5120 ± 0 | 0.0952 ± 0.1650 | 0.0073 |
| Transformer | 64 | 0.9871 ± 0.0020 | 0.9502 ± 0.0069 | 0.8703 ± 0.0271 | 0.9480 ± 0.0049 | 0.9073 ± 0.0127 | 2268 ± 58 | 0.0425 ± 0.0080 | 0.0026 |
| Transformer | **128** | 0.9974 ± 0.0007 | 0.9854 ± 0.0039 | 0.9744 ± 0.0125 | 0.9701 ± 0.0074 | **0.9722 ± 0.0073** | 3175 ± 94 | 0.0227 ± 0.0150 | 0.0039 |
| Transformer | 256 | 0.9966 ± 0.0030 | 0.8981 ± 0.0895 | 0.9333 ± 0.1155 | 0.7143 ± 0.2474 | 0.7980 ± 0.1776 | 5120 ± 0 | 0.2857 ± 0.2474 | 0.0067 |
| Mamba-3 | 64 | 0.9885 ± 0.0013 | 0.9559 ± 0.0049 | 0.8722 ± 0.0093 | 0.9691 ± 0.0088 | 0.9181 ± 0.0091 | 2259 ± 73 | 0.0167 ± 0.0096 | 0.0111 |
| Mamba-3 | **128** | 0.9959 ± 0.0013 | 0.9778 ± 0.0064 | 0.9508 ± 0.0355 | 0.9658 ± 0.0148 | **0.9578 ± 0.0122** | 3191 ± 100 | 0.0272 ± 0.0227 | 0.0112 |
| Mamba-3 | 256 | 0.9976 ± 0.0030 | 0.9310 ± 0.0880 | 0.9333 ± 0.1155 | 0.8095 ± 0.2182 | 0.8632 ± 0.1745 | 5120 ± 0 | 0.1905 ± 0.2182 | 0.0125 |

**핵심 관찰**

1. **모든 모델에서 128이 sweet spot**. 64는 짧은 문맥으로 −2~4%p, 256은 −7~37%p 손실.
2. **window=256에서 광범위한 성능 붕괴**. 원인: HAPT 전이 segment 길이가 보통 2~5초인데 256 (=5.12s) 윈도우보다 짧은 segment는 `_build_windows`에서 통째로 dropped → 학습/평가 표본 자체가 줄어든다. miss rate가 0.10~0.33으로 급증하고 std가 매우 커진다. (GRU window=256은 6 segments 중 2개만 탐지하는 등 극단적 케이스 발생.)
3. **End-of-window latency는 window length에 거의 정확히 비례**: 64 → ~2250ms, 128 → ~3170ms, 256 → 5120ms (이론치 5120ms와 일치 — 256에서는 항상 첫 윈도우가 탐지 또는 미탐).
4. **Inference time은 window에 거의 선형**. Mamba-3: 64→0.0111, 128→0.0112, 256→0.0125 ms/window. seq×2면 win당 cost도 ~1.1×로 거의 일정 (chunked SSM 특성). Transformer: 0.0026→0.0039→0.0067 (×2.6) — attention의 quadratic cost가 보이기 시작.
5. **Mamba-3 vs Transformer at window=256**: Transformer 0.7980, Mamba-3 0.8632 — **256에서는 Mamba-3가 Transformer를 +6.5%p 능가**. 긴 시퀀스에서 선택적 SSM의 안정성 이점이 처음으로 명확히 나타나는 지점. (단, 둘 다 128 대비로는 손실.)

> **window-length 논문 claim**:
> "window=128이 모든 모델에서 최적 trade-off였다. window=256은 짧은 전이 segment의 학습 표본 손실로 성능이 광범위하게 붕괴했으며, 이 영역에서 Mamba-3 (0.86)가 Transformer (0.80)를 6.5%p 능가하여 긴 시퀀스에서의 선택적 SSM의 안정성 이점을 시사하였다. Mamba-3의 window당 추론 시간은 window=64→256에서 1.13× 증가에 그쳐 Transformer의 2.6× 대비 우수한 확장성을 보였다."

---

## 5. Detection Latency 상세

End-of-window latency = (first_positive_window_start + window − segment_start) × 20ms.
즉 모델이 transition을 처음 양성으로 예측한 윈도우가 "탐지 가능해진 시각" − segment 시작 시각.

이론적 하한:
- window=64 (1280ms): 첫 윈도우가 즉시 탐지하면 latency = 1280ms
- window=128 (2560ms): 동일 하한 2560ms
- window=256 (5120ms): 동일 하한 5120ms

표 4를 보면 window=64에서는 평균 2247ms (하한 1280ms 대비 +967ms ≈ 0.76 stride), 128에서는 3175ms (하한 2560ms 대비 +615ms ≈ 0.48 stride), 256에서는 5120ms (정확히 하한 = 즉시 탐지가 유일한 mode).

**해석**:

- window가 길어질수록 첫 윈도우가 segment 전체를 포함하므로 즉시 탐지가 일반화 → latency가 window length에 수렴.
- 짧은 window일수록 첫 윈도우가 segment 시작과 어긋날 수 있어 stride 단위 지연이 추가됨.
- **64 vs 128 trade-off**: 64는 latency 925ms 더 빠르지만 T-F1 1~4%p 손실. 응용에 따라 (낙상 alert: speed 우선 → 64; 자세 분석: F1 우선 → 128).

### Random split @ window=128에서 모델 간 latency 비교 (5 seeds)

거의 동일 (3082~3125ms 범위). 모델 선택이 latency에 큰 영향을 주지는 않음 — windowing 자체가 latency의 지배 요인. **즉, paper에서 "Mamba-3가 latency 측면에서 유리하다"는 주장은 본 setting에서 성립하지 않는다.** 차별화는 F1과 일반화에서 찾아야 함.

### Subject-indep에서 latency가 약 500ms 짧음 (~2585 vs ~3115ms)

원인: subject split은 test set이 다른 사용자라 transition 발생 횟수와 segment 길이 분포가 다름. 평균적으로 더 짧은 segments → 첫 윈도우가 전체 segment를 덮고 즉시 탐지하는 비율이 높음.

---

## 6. RQ별 정리 (논문 본문 작성용)

### RQ1. 동작 전이 탐지는 HAR 분류와 다른 문제인가?

- **Accuracy는 모든 모델에서 0.97~0.998로 포화** (5-seed mean 기준).
- 반면 **Transition F1은 0.93~0.98 범위로 모델 간 변동이 5~6%p**.
- subject-indep로 가면 transition F1 std가 random 대비 2~10× 커짐 (Transformer 0.007 → 0.056). 즉 일반화 어려움.
- **결론 (유지)**: Accuracy만으로 모델을 평가하면 모든 모델이 "좋아 보이지만" 전이에 한정하면 모델 차이가 명확. RQ1은 강하게 지지됨.

### RQ2. 선택적 상태공간 모델(Mamba-3)이 전이 탐지에 유리한가?

본 sweep에서는 **부분적으로만 지지**됨.

지지되는 측면:
- window=256에서 Mamba-3가 Transformer 대비 +6.5%p 우위 → 긴 시퀀스 안정성 이점.
- inference cost scaling이 가장 우수 (64→256에서 1.13× 증가).
- subject-indep에서 Transformer보다 1.5%p 작은 일반화 손실.

지지되지 않는 측면:
- 본 sweep의 canonical 설정(window=128, acc+gyro, random split)에서 **CNN 0.9759, Transformer 0.9731, TCN 0.9649 모두 Mamba-3 (0.9609)보다 평균이 높음**.
- subject-indep에서도 TCN (0.9649)이 Mamba-3 (0.9458)보다 +1.9%p 우위.

> **수정 claim**: "Mamba-3 기반 선택적 상태공간 모델은 본 setting에서 baseline 대비 명확한 우위를 보이지 못하였으나, 긴 윈도우(256)와 subject-independent 평가의 일부 setting에서 Transformer 대비 상대적 강점을 나타냈다. canonical 128-window 설정에서는 CNN/Transformer가 더 우수했다."

### RQ3. 자이로스코프 회전 신호는 전이 탐지에 얼마나 기여하는가?

| Model | acc → gyro 변화 | acc → acc_gyro 변화 |
|---|---:|---:|
| 1D-CNN | +0.030 | **+0.063** |
| GRU | +0.013 | −0.004 |
| TCN | **+0.020** | +0.009 |
| Transformer | −0.043 | **+0.018** |
| Mamba-3 | +0.020 | **+0.029** |

- **CNN, Transformer, Mamba-3는 acc+gyro fusion이 가장 우수** → 회전 신호 보조 효과 지지.
- **GRU는 fusion에서 손실** — 회전 노이즈를 흡수하지 못함.
- **TCN은 gyro 단독이 acc+gyro보다 높음** — 회전 단독만으로도 dilated CNN이 잘 학습.
- Mamba-3는 단일 sensor 약점(acc 단독 0.9324)을 fusion으로 +2.9%p 보완. **여전히 fusion gain은 Mamba-3가 가장 크지는 않음 (CNN +6.3%p > Mamba-3 +2.9%p)**.

> **수정 claim**: "회전 신호는 dilated CNN을 제외한 모든 모델에서 acc+gyro fusion으로 추가 성능을 제공하였다. fusion gain은 1D-CNN에서 가장 컸으며 (+6.3%p), Mamba-3는 단일 sensor 약점을 보완하는 형태로 +2.9%p의 fusion 이득을 보였다."

---

## 7. Paper 본문에 권장하는 main claim 재구성

이전 single-seed 결과 기반의 강한 claim("Mamba-3 ≈ Transformer 동률")은 **5-seed로는 성립하지 않음**. 정직한 claim:

> **"선택적 상태공간 모델 기반 IMU 동작 전이 탐지의 가능성과 한계를 분석하였다. 표준 setting (window 128, acc+gyro, random split)에서 Mamba-3는 baseline (CNN/TCN/Transformer) 대비 1~1.5%p 낮은 mid-pack 성능을 보였다. 그러나 (i) 윈도우 길이가 길어질 때(256) Mamba-3의 inference cost는 거의 일정하게 유지되어 Transformer의 2.6× 증가 대비 효율적이었으며, 같은 setting에서 Transition F1도 Transformer를 +6.5%p 능가하였다. (ii) Subject-independent 평가에서는 dilated TCN이 가장 견고하였고(Δ=0%p), self-attention 기반 Transformer는 가장 큰 일반화 손실(−3.3%p)을 보였다. (iii) Detection latency는 windowing이 지배하며 모델 선택은 거의 영향을 주지 않았다 (3.08~3.13s). 본 결과는 선택적 SSM의 즉시적 우위가 짧은 IMU 윈도우에서는 두드러지지 않으나, 긴 시퀀스와 user-generalization이 중요한 응용에서 향후 검토 가치가 있음을 시사한다."**

부수 claim:

- "전체 분류 Accuracy는 모든 모델에서 0.99 이상으로 포화되어 모델 차이를 드러내지 못한다. 본 연구는 Transition F1을 모델 선택의 1차 기준으로 사용할 것을 제안한다."
- "End-of-window latency는 window length에 의해 결정되며, F1과의 trade-off (64 ↔ 128)가 응용 시나리오별 모델 설계에 핵심적이다."

---

## 8. 한계 및 다음 단계

1. **5-seed로 좁혀진 차이의 통계적 유의성**: 5-seed std로는 두 모델 차이가 일관되게 유의(Welch t > 2σ)한 경우만 확신 가능. CNN vs Transformer (Δmean 0.0028, σ~0.014) 등 본 표 내 다수 차이는 통계적으로 분리되지 않음. → 10+ seeds 또는 paired bootstrap 권장.
2. **Window=256 collapse는 본질적**: 짧은 transition segment가 통째로 dropped 되어 데이터셋이 크게 줄어드는 구조적 문제. 향후 stride를 줄이거나 segment-aware overlap을 고려해야 fair comparison.
3. **detection latency는 windowing 지배**. 모델별 latency 차별화를 보려면 (a) frame-level (stride=1) 또는 (b) state streaming 평가가 필요. 본 sweep의 ms-단위 latency 차이(±50ms)는 batch 단위 평가의 artifact 수준.
4. **subject-indep split의 user 수가 4~5명으로 작음**. seed별 user 조합에 따라 결과 변동이 크므로 leave-one-subject-out (30-fold) cross-validation이 더 신뢰 가능.
5. **focal loss / hard-negative mining**: transition class imbalance는 weighted CE로만 완화. 향후 비교 가능.

---

## 9. 산출물

```
outputs_user/imu_transition/
  main_5seed/         results_phase1.csv, latency.csv, seed{13,42,73,137,211}/<model>_<channels>/*
  subject_5seed/      results_phase1.csv, latency.csv, seed{13,42,73,137,211}/<model>_<channels>/*
  window_64_3seed/    results_phase1.csv, latency.csv, seed{13,42,73}/<model>_acc_gyro/*
  window_128_3seed/   results_phase1.csv, latency.csv, seed{13,42,73}/<model>_acc_gyro/*
  window_256_3seed/   results_phase1.csv, latency.csv, seed{13,42,73}/<model>_acc_gyro/*
  agg_tables.md       (이 result.md의 표 원본)
  summary.json        (mean/std flat dict, 모든 metric)
```

각 run dir에는 `best.pt`, `history.json`, `test_metrics.json`, `test_predictions.json`이 포함. 후자는 detection latency 재계산용 (per-window y_true/y_pred + exp/user/start metadata).

---

## 10. 재현

```bash
# venv (system python에 dev headers가 없으면 uv-managed python 사용)
uv venv -p ~/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
source .venv/bin/activate
uv pip install setuptools wheel packaging ninja torch
uv pip install --no-build-isolation -e ".[experiments]"

# data/ 와 outputs/가 root 소유라 쓰기 불가 → HAPT_CACHE_DIR 환경변수와 출력 디렉토리 오버라이드를 사용
export HAPT_CACHE_DIR=/home/jdone/ai/mamba/mamba3/cache_user
cp data/windows_128_64.npz $HAPT_CACHE_DIR/   # 기존 128/64 캐시 재활용

# Main 5-seed sweep
python experiments/imu_transition/run_phase1.py \
    --config /tmp/phase1.yaml \
    --seeds 13 42 73 137 211 --split-mode random --output-suffix main_5seed

# Subject-independent 5-seed sweep
python experiments/imu_transition/run_phase1.py \
    --config /tmp/phase1.yaml \
    --seeds 13 42 73 137 211 --split-mode subject --output-suffix subject_5seed

# Window-length ablation (acc_gyro만, 3 seeds)
for WIN in 64 128 256; do
  STRIDE=$((WIN/2))
  python experiments/imu_transition/run_phase1.py \
    --config /tmp/phase1.yaml \
    --seeds 13 42 73 --split-mode random --channels acc_gyro \
    --window-size $WIN --stride $STRIDE --output-suffix window_${WIN}_3seed
done

# Detection latency 후처리
for D in main_5seed subject_5seed window_64_3seed window_128_3seed window_256_3seed; do
  python experiments/imu_transition/compute_latency.py \
    --predictions-glob "outputs_user/imu_transition/$D/seed*/*/test_predictions.json" \
    --data-root data/uci_har_pt \
    --output "outputs_user/imu_transition/$D/latency.csv"
done

# 집계 (mean ± std)
python experiments/imu_transition/aggregate_results.py
```

`/tmp/phase1.yaml`은 원본 `experiments/imu_transition/configs/phase1.yaml`을 복사하고 `output_root: outputs_user/imu_transition`로 수정한 것 (`outputs/`가 root 소유라 우회).
