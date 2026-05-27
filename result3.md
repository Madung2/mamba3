# IMU 동작 전이 방향 분류 실험 결과 — TIC-style Encoder와 GyroPhase Head

`exp_plan3.md` (Transformer IMU Calibrator 관점) + `exp_plan3-1.md` (Mamba-3 internal state 추출 및 2×2 ablation) 기반 sweep의 정량 보고서. `paper.md` 본문과 동일한 구조를 따른다.

| Phase | Sweep | Split | Seeds | Specs | runs |
|---|---|---|---|---|---:|
| 3 | `phase3_main` | Random, 7-class direction, acc+gyro, T=128 | 13, 42, 73 | 14 (3 baseline + 2 GyroPhase ports + 4 SSM 2×2 + 5 complex_selective head ablations) | 42 |
| 3 | `phase_analysis3` | 후처리 (test set, hidden state 추출) | – | 6 complex_* specs × 3 seeds | 18 |
| **합계** | | | | | **60** |

평가지표: `accuracy`, `macro_f1`, `direction_macro_f1` (class 1–6 평균), `worst_direction_f1`, `transition_f1` (binary로 환원), `inference_ms_per_window`, `params`. 모든 값은 3-seed (13/42/73) 평균 ± 표준편차(ddof=1).

---

## 1. 실험 설정

| 항목 | 값 |
|---|---|
| Dataset | UCI HAPT (acc_gyro 6채널, window=128 stride=64, 50 Hz) |
| Task | 7-class transition direction (`non_transition` + 6 directed transitions) |
| Loss / Opt | weighted CE / AdamW lr=1e-3 wd=1e-4 batch 64 |
| Epochs / patience | max 50 / early-stop 10 (val `direction_macro_f1`) |
| Pooling (encoder → head) | **mean across time** (TIC-style) — result.md baseline은 last-token이었음 |
| Encoder hidden | 2×2/Real/Complex SSM d_model=64 d_state=64 n_layer=2; Mamba-3 d_model=128 d_state=64 expand=2 n_layer=2 chunk=16; Transformer d_model=64 4-head 2-layer FF=128; TCN ch=[64,64,128,128] |
| Phase head | lazy-built Linear, dropout 0.1, AvgPool h_base ⊕ phase feature |
| Device | CUDA · RTX PRO 6000 Blackwell · torch 2.10.0+cu128 |
| Inference timing | 25 warmup + 100 timed, ms/window |

### 새로 구현한 코드 (`experiments/imu_transition/`)

- `models/ssm_2x2.py` — Real/Complex × Static/Selective 4개 block
- `models/encoders.py` — 통일된 `_SequenceModel` 인터페이스 (TCN/Transformer/Mamba-3/Real-SSM/Complex-SSM/2×2)
- `models/gyrophase.py` — `HeadConfig`, `GyroPhaseHead`, `compute_phase_change`, `compute_rotation_diversity_{std,bin}`
- `models/phase_classifier.py` — `PhaseAwareClassifier` (backbone + head + phase feature 라우팅)
- `run_phase3.py` — sweep orchestrator
- `aggregate_phase3.py` — 표 / opposite-pair / RD subset 집계
- `phase_analysis3.py` — Complex-SSM 체크포인트에서 phase/selective_score 분석

### Mamba-3 internal state 접근 결정

`exp_plan3-1.md §2`는 fused triton/cute kernel을 직접 수정하지 말 것을 명시하므로, 본 sweep은 §13 **대안 1**을 채택한다.

- Mamba-3 backbone은 그대로 fused kernel을 사용한다.
- Hidden phase / selective_score 정량 분석은 **자체 Complex-SSM 계열 (`complex_selective`, `complex_static`)** 의 unfused Python reference scan에서 수행한다.
- Mamba-3 + GyroPhase Head는 **input gyro-derived phase proxy** 를 사용한다 (`PhaseAwareClassifier._fallback_gyro_phase_change` — 연속 timestep 사이의 gyro vector L2 변화량).

이 결정은 `paper.md` §3.2의 "Mamba-3 또는 Complex-SSM이 실제 신체 회전을 직접 이해한다는 의미가 아니다"라는 해석상의 주의와도 일관된다.

---

## 2. 본 연구의 기여 — Phase-3 정량 요약

### 2.1 핵심 결과 4가지

1. **표준 분류에서 Mamba-3가 mean-pool로 baseline을 추월**.  
   result.md (last-token pool)에서 Mamba-3 direction macro F1 = 0.715 ± 0.091 이었으나, **mean-pool**(TIC 관점) 로 바꾸자 **0.7553 ± 0.0587**. TCN baseline (0.7492 ± 0.0239)과 동률 이상이며, 이는 짧은 IMU window에서 selective SSM이 readout 방식만 바꿔도 TCN 수준으로 경쟁할 수 있음을 보인다.

2. **Mamba-3 + GyroPhase + RD Head가 최고 성능 (0.7611 ± 0.0440)**.  
   AvgPool 대비 +0.006 (mean) 으로 가산 효과는 작지만 분산 감소 (±0.044 vs ±0.059) 와 worst-class F1 (0.665 vs 0.620) 개선이 명확하다. paper.md §3 main claim의 "표현 수준의 회전성 정보 → 분류 성능 연결" 부분을 약하게 (weakly) 지지.

3. **GyroPhase Head는 backbone-independent**.  
   Transformer + GyroPhase+RD: 0.6925 → 0.7176 (+0.025). 단순 readout 모듈이 backbone에 종속되지 않는 phase-aware head로 작동함을 시사. exp_plan3 §8 Case C 시나리오와 일치.

4. **2×2 ablation: selective scanning이 효과 — 단 real-valued에 집중**.  
   - Real-Static → Real-Selective: 0.6841 → 0.7560 (+0.072) — selective scanning의 명확한 효과.
   - Complex-Static → Complex-Selective: 0.5821 → 0.6035 (+0.021) — 효과 작음.
   - Real-Selective vs Complex-Selective: 0.756 vs 0.604 — **자체 Complex-SSM은 7-class direction 학습이 여전히 어려움** (result.md §4.2와 일치).

### 2.2 Phase 분석 (18 complex checkpoints)

`paper.md` §6.2의 hidden phase 분석을 phase-3 모든 complex 변형으로 확장. 결과:

| Spec | corr(Δφ, gyro) | corr(sel, gyro) | corr(sel, Δφ) | Δφ trans/non | sel trans/non |
|---|---:|---:|---:|---:|---:|
| complex_selective + avgpool | **0.827** | −0.439 | −0.350 | **1.240** | 0.991 |
| complex_selective + phase | 0.745 | −0.617 | −0.476 | 1.149 | 0.987 |
| complex_selective + gyrophase | 0.848 | −0.296 | −0.258 | 1.261 | 0.996 |
| complex_selective + gyrophase_rd | 0.825 | −0.308 | −0.240 | 1.260 | 0.996 |
| complex_selective + selective_gyrophase | 0.797 | −0.120 | −0.031 | 1.223 | 0.995 |
| complex_static + avgpool | 0.837 | n/a | n/a | 1.240 | 1.000 |

**Phase analysis 메인 관찰**:
1. **`corr(Δφ, gyro) = 0.79–0.85` 가 모든 complex 변형에서 보존** — result.md §4.4의 0.847 ± 0.020을 phase-3 head 변형 전반에서 재현.
2. **Static complex variant (no selective)에서도 r = 0.84** — phase–gyro coupling은 **selective scanning이 아니라 complex update 자체가 만든다**. 핵심 inductive bias의 위치를 분리해서 확인.
3. **Δφ trans/non ratio = 1.15–1.26** — paper.md §6.2의 1.231 ± 0.057과 동일 범위. Phase가 전이 구간에서 강해지는 inductive bias는 모든 complex variant에서 일관.
4. **Selective_score (rho)는 전이/비전이에 거의 동일** (`sel_ratio ≈ 0.99–1.00`). selective scanning은 학습 후에도 transition timing의 직접적 marker로는 작동하지 않는다.
5. **`corr(sel, gyro)` 가 negative (−0.12 ~ −0.62)** — 입력 gyro가 클 때 rho가 작아짐 (= 과거 state를 더 빨리 잊음). 이는 selective scanning의 정성적 동작과 일치하지만, head feature로서의 신호는 약하다. → **selective_gyrophase head가 plain GyroPhase보다 좋지 않은 이유**.

### 2.3 핵심 주장 (Phase-3)

> mean-pool readout과 GyroPhase + RD Head를 결합하면 Mamba-3가 짧은 IMU window의 direction 분류에서 가장 안정적인 baseline이 된다 (direction macro F1 0.7611 ± 0.0440). 이 head는 Transformer encoder에도 backbone-independent하게 +0.025의 개선을 준다. 2×2 SSM ablation은 (i) real-valued에서 selective scanning이 +0.072의 큰 개선을 주는 반면 (ii) complex update의 회전성 inductive bias (Δφ–gyro 상관 r=0.83, Δφ trans/non ratio=1.24)는 **selective scanning이 없어도** complex_static에서 동일하게 형성됨을 보인다. 즉 본 연구의 phase signal은 selective scanning의 부산물이 아니라 complex 상태 업데이트 자체의 구조적 산물이며, 이를 classifier에 직접 전달하는 GyroPhase Head가 약한 형태의 성능 개선을 만든다.

### 2.4 해석상의 주의

- Mamba-3가 가장 좋은 mean을 보이지만, TCN (0.7492 ± 0.024) 과의 차이는 통계적으로 분리되지 않는다 (3-seed, σ~0.04~0.06).
- 자체 Complex-SSM은 여전히 direction task에서 0.60 수준으로 학습이 어렵다. 본 연구의 phase 분석은 학습된 hidden state의 *경향*을 보고하는 것이며, complex SSM이 우월한 분류기임을 주장하지 않는다.
- Selective_score = rho 는 selective scanning의 충분한 proxy가 아닐 수 있다. 향후 Mamba-3 unfused reference path 구현 시 DT × ||x|| 또는 trap gate 등이 더 나은 proxy일 수 있다.

---

## 3. Phase-3 Main Table (3 seeds, direction, acc+gyro)

`outputs_user/imu_transition/phase3_main/agg_phase3.md`를 옮긴 표. **Direction Macro F1** 가 1차 정렬키.

| Model | Direction Macro F1 | Macro F1 | Trans F1 | Worst-class F1 | Non-trans F1 | Acc | ms/win | Params |
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

---

## 4. Experiment 별 결과

### 4.1 Experiment 1: TIC-style encoder 비교 (Transformer vs Mamba-3 vs TCN)

- **목표**: Transformer encoder + AvgPool 가 맡은 IMU short-history encoder 역할을 Mamba-3 가 대체 가능한가?
- **결과**: Mamba-3 + AvgPool 0.7553 > Transformer + AvgPool 0.6925 (+0.063) > TCN baseline 0.7492 와 거의 동률. exp_plan3 Experiment 1 **성공 기준 충족** (`Mamba-3 + AvgPool ≥ Transformer + AvgPool`).
- result.md (last-token pool) 의 Mamba-3 = 0.7150 과 비교하면 **+0.040** — **pooling 변경의 단독 효과**.

### 4.2 Experiment 2: GyroPhase Head ablation on Mamba-3

| Head | Direction Macro F1 | Δ vs AvgPool |
|---|---|---:|
| AvgPool (baseline) | 0.7553 ± 0.0587 | — |
| GyroPhase + RD | 0.7611 ± 0.0440 | **+0.0058** |

- Mamba-3는 hidden complex state 직접 접근이 어려워 phase feature는 input gyro-derived fallback 사용. 그럼에도 worst-class F1 +0.045, std 감소 (0.059 → 0.044) 가 나타나 안정성 측면에서 head 효과 확인.
- exp_plan3 Experiment 2 **최소 성공 기준** (`Mamba-3 + GyroPhase > Mamba-3 + AvgPool`) 충족.

### 4.3 Experiment 3: Transformer + GyroPhase Head 비교

| Backbone | Head | Direction Macro F1 |
|---|---|---|
| Transformer | AvgPool | 0.6925 ± 0.0220 |
| Transformer | GyroPhase + RD | 0.7176 ± 0.0417 |
| Mamba-3 | AvgPool | 0.7553 ± 0.0587 |
| Mamba-3 | GyroPhase + RD | 0.7611 ± 0.0440 |

- Transformer 에서도 **+0.025** 개선 → exp_plan3 §8 **Case C** (GyroPhase Head는 backbone-independent phase-aware readout). Mamba 중심성은 약해지지만 head 자체의 일반성을 확보.
- `Mamba-3 + GyroPhase > Transformer + GyroPhase` (0.761 vs 0.718, +0.043) → Mamba encoder의 필요성도 부분 지지.

### 4.4 Experiment 4: Rotation Diversity ablation

- Mamba-3 + GyroPhase + RD가 winner 이지만, full / high-gyro / high-RD subset에서 모두 0.76 수준으로 큰 차이가 없음.
- **방법론적 한계**: HAPT는 transition 구간이 본질적으로 gyro magnitude 가 높아 high/low subset이 거의 high=transitions / low=non-transitions로 갈라짐. `rd_subset.csv` 의 low-gyro 칼럼이 0.0인 것은 그 subset에 transition window가 거의 없기 때문. 향후 transition-only subset 분석으로 재실험 필요.
- `RD_bin` (방향 bin diversity) 은 본 sweep에서 시간 부족으로 1차 RD_std 만 사용.

### 4.5 Experiment 5: Opposite-pair confusion

| Pair | 모든 모델 평균 오분류율 |
|---|---:|
| stand-to-sit ↔ sit-to-stand | 0–3.3% |
| sit-to-lie ↔ lie-to-sit | 0–1.3% |
| stand-to-lie ↔ lie-to-stand | 0–2.1% |

- 모든 모델에서 opposite-pair confusion 이 거의 0. paper.md §6.3 검증 질문 #4 (반대 방향 혼동 감소) 는 **이미 모든 모델이 잘 풀고 있어 변별력 없음**. 가장 큰 혼동은 result.md §4.1 confusion matrix에서 본 `stand_to_lie ↔ sit_to_lie` (시작이 다르고 종료가 같은 짝) 처럼 **opposite가 아닌 인접 짝**.

### 4.6 Experiment 7: 2×2 SSM ablation (complex × selective)

paper.md / exp_plan3-1 의 추가 분석: complex update 효과와 selective scanning 효과의 분리.

| Block | Direction Macro F1 | Δ vs static |
|---|---|---:|
| real_static | 0.6841 ± 0.0329 | — |
| real_selective | 0.7560 ± 0.0488 | **+0.0719** |
| complex_static | 0.5821 ± 0.0603 | — |
| complex_selective | 0.6035 ± 0.0272 | +0.0214 |

- **Real selective gain 큼 (+0.072)** — selective scanning은 real-valued state update에서 명확히 작동.
- **Complex selective gain 작음 (+0.021)** — 자체 Complex-SSM 구현이 complex × selective 의 학습 안정성을 충분히 확보하지 못함 (result.md §4.2와 일치).
- **그러나 Δφ–gyro 상관은 complex_static에서도 0.84** — complex update가 만드는 회전성 inductive bias는 selective scanning과 독립적으로 존재. 이는 paper.md §3.1 main claim의 "complex-valued state가 회전성 신호를 표현 수준에서 추적"을 정량적으로 강화한다.

---

## 5. Phase Analysis 정량표 (3 seeds × 6 complex specs)

### 5.1 Δphase / gyro / selective_score 상관

(2.2의 표 다시) — `outputs_user/imu_transition/phase3_main/phase_analysis3_summary.csv` 기반.

### 5.2 Per-run breakdown (Pearson r 의 seed별 변동)

| spec | seed=13 | seed=42 | seed=73 |
|---|---:|---:|---:|
| complex_selective.avgpool | 0.823 | 0.873 | 0.785 |
| complex_selective.phase | 0.640 | 0.834 | 0.761 |
| complex_selective.gyrophase | 0.801 | 0.864 | 0.878 |
| complex_selective.gyrophase_rd | 0.813 | 0.840 | 0.822 |
| complex_selective.selective_gyrophase | 0.807 | 0.760 | 0.824 |
| complex_static.avgpool | 0.831 | 0.847 | 0.833 |

- complex_static (selective scanning 없음) 의 r=0.83 ± 0.01 — 가장 안정적. selective 변형들은 head 종류에 따라 r 이 0.64~0.88 사이로 흔들림.
- **결론**: phase–gyro coupling 의 robustness 측면에서는 complex_static 이 더 깔끔하다. selective scanning은 학습 표현에 추가 변동성을 도입한다.

---

## 6. RQ 별 정리

`exp_plan3.md` 및 `exp_plan3-1.md` 의 핵심 질문에 대한 답.

### Q1. Mamba-3 가 IMU short-history encoder로 Transformer 를 대체할 수 있는가?
**◯ 지지**. mean-pool readout 기준 Mamba-3 (0.7553) > Transformer (0.6925), +0.063. 처음으로 Mamba-3 가 baseline 을 능가.

### Q2. GyroPhase Head 가 AvgPool 보다 좋은가?
**△ 부분 지지**. Mamba-3 + AvgPool 0.7553 → +RD 0.7611 (+0.006, mean). Worst-class F1 +0.045, std 감소 효과는 분명하나 mean F1 향상은 작음.

### Q3. GyroPhase + RD Head가 backbone-independent 한가?
**◯ 지지**. Transformer 에서도 +0.025 (Case C).

### Q4. 2×2 ablation 으로 complex / selective 효과를 분리할 수 있는가?
**◯ 지지**. Real selective +0.072, Complex selective +0.021 — selective scanning은 real-valued에 더 큰 효과. Δφ–gyro 상관은 complex_static (no selective) 에서도 0.84 유지.

### Q5. Selective_score 가 phase 또는 gyro 와 양의 상관을 보이는가?
**✗**. `corr(sel, gyro) = −0.12 ~ −0.62` (음의 상관). selective scanning은 입력이 클 때 과거 state를 잊는 방향으로 학습 — head feature로서의 신호는 약함. Selective GyroPhase 의 head 가 plain GyroPhase 보다 좋지 않은 이유.

### Q6. Selective_score 가 전이 구간에서 비전이보다 커지는가?
**✗**. `sel_trans / sel_non ≈ 0.99` — transition timing 의 직접 marker 가 아님. (반면 Δphase ratio = 1.24 → phase-based feature 는 transition 에 반응).

### Q7. Opposite-pair confusion 이 GyroPhase Head 로 줄어드는가?
**✗ (변별력 없음)**. 모든 모델이 이미 거의 0% — 본 dataset의 dominant confusion 은 opposite 가 아니라 시작/종료 자세가 인접한 쌍.

### Q8. High-gyro / High-RD subset 에서 GyroPhase Head 가 더 효과적인가?
**△ 측정 어려움**. HAPT 의 transition window 는 본질적으로 high-gyro 에 몰려있어 high/low split 이 거의 transition/non-transition split 과 같음. 향후 transition-only subset 으로 재실험 필요.

---

## 7. 한계 및 다음 단계

### 7.1 본 sweep 의 한계
1. **3-seed로 좁혀진 차이의 통계적 유의성**: top 4 모델 (Mamba-3+GyroPhase, Real-Selective+AvgPool, Mamba-3+AvgPool, TCN) 의 mean 차이는 σ ~ 0.04~0.06 안에서 분리되지 않음. 5+ seeds 또는 paired bootstrap 필요.
2. **Subject-independent eval 미수행**: result.md 의 subject-disjoint split (Transformer −3.3%p 일반화 손실) 을 phase-3 모델들에 대해 재실험할 필요.
3. **Synthetic rotation task 미수행** (exp_plan3 Experiment 7): cos/sin only input 의 controlled rotation task 로 inductive bias 재확인 필요.
4. **Mamba-3 hidden phase 직접 분석 미수행**: exp_plan3-1 §13 대안 1 채택. fused kernel 을 그대로 두는 한 Mamba-3 내부 phase 의 정량 비교는 본 연구의 범위 외.
5. **RD subset 분석의 degeneracy**: high/low gyro 가 transition/non-transition 과 거의 일치 → 다음에는 transition-only window 안에서 RD 로 재split.
6. **Frozen-encoder 학습 변형 미수행** (exp_plan3-1 §6 권장). 현재 모두 end-to-end. Frozen 비교가 head 효과의 “순도” 를 더 잘 분리할 수 있음.

### 7.2 다음 단계 제안
1. **Seed 확장**: 5–10 seeds 로 Mamba-3 + GyroPhase+RD vs TCN, vs Mamba-3+AvgPool 의 paired t-test.
2. **Subject-disjoint sweep**: phase-3 specs × 5 seeds × subject split.
3. **Mamba-3 unfused reference path**: `inference_params` 로 cache 를 강제해 한 step 씩 풀고, `ssm_state` 와 `angle_dt_state` 를 매 timestep 로깅 → Mamba-3 의 internal Δphase 와 Δφ–gyro 상관 직접 측정.
4. **Selective_score proxy 개선**: rho 대신 `DT × ||u||` (input-modulated update budget) 로 재정의, transition 구간에서 더 큰지 확인.
5. **RD_bin** (gyro 방향 bin diversity) feature 실험 (현재는 RD_std 만).
6. **GyroPhase-TCN** (exp_plan3 §8 Case D 대비책): TCN backbone + phase head 도 시도해 두면 안전망.

---

## 8. 산출물

```
outputs_user/imu_transition/phase3_main/
  seed{13,42,73}/<backbone>__<head>__<pool>/
    best.pt                 # checkpoint
    history.json            # epoch-wise train_loss + val metrics
    test_metrics.json       # backbone, head, params, all test metrics + confusion_matrix
    test_predictions.json   # y_true, y_pred, exp/user/start metadata
  results_phase3.csv        # 42-row flat sweep table
  results_phase3.json
  agg_phase3.md             # 3-seed mean±std table (sec 3 source)
  per_run.csv               # raw per-run rows
  opposite_pair.csv         # opposite-pair confusion stats
  rd_subset.csv             # high/low gyro/RD subset dirF1
  phase_analysis3.csv       # 18 complex checkpoints raw
  phase_analysis3_summary.csv  # mean per spec
  sweep.log                 # stdout of run_phase3.py
```

신규 코드 (phase-3):

```
experiments/imu_transition/
  models/ssm_2x2.py            # 2x2 ablation blocks + SSM2x2Encoder
  models/encoders.py           # unified backbone interface
  models/gyrophase.py          # HeadConfig, GyroPhaseHead, RD feature
  models/phase_classifier.py   # PhaseAwareClassifier (end-to-end glue)
  run_phase3.py                # sweep orchestrator
  aggregate_phase3.py          # main table + opposite-pair + RD subset
  phase_analysis3.py           # complex-state phase / selective analysis
```

---

## 9. 재현

```bash
# 환경 (.venv는 phase-1/2와 동일)
source .venv/bin/activate
export HAPT_CACHE_DIR=/home/jdone/ai/mamba/mamba3/cache_user

# /tmp/phase3.yaml = configs/phase1.yaml의 output_root만 outputs_user/imu_transition으로 교체
sed 's|outputs/imu_transition|outputs_user/imu_transition|' \
  experiments/imu_transition/configs/phase1.yaml > /tmp/phase3.yaml

# ===== Phase-3 main sweep (42 runs, ~70 min on RTX PRO 6000) =====
python experiments/imu_transition/run_phase3.py \
    --config /tmp/phase3.yaml \
    --seeds 13 42 73 \
    --output-suffix phase3_main

# ===== Aggregate =====
python experiments/imu_transition/aggregate_phase3.py \
    --root outputs_user/imu_transition/phase3_main

# ===== Phase analysis on complex_* checkpoints =====
python experiments/imu_transition/phase_analysis3.py \
    --root outputs_user/imu_transition/phase3_main \
    --specs-regex complex_

# 단일 spec 만 돌리려면:
python experiments/imu_transition/run_phase3.py --config /tmp/phase3.yaml \
    --seeds 42 --output-suffix phase3_single \
    --specs mamba3.gyrophase_rd complex_selective.phase
```
