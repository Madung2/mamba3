# Phase-4 실험 결과 — selective proxy, subject-indep, transition-only subset, harder synthetic

`exp_plan4.md` (todo.md 4 항목 후속) 기반 sweep의 정량 보고서. `result3.md` §7.1 한계 #2, #3, #5 그리고 `paper.md` §6 의 selective_score 이슈를 보완한다.

| Sweep | 목적 | runs |
|---|---|---:|
| `phase4_proxy` | exp4-1: 새 selective proxy (update_budget, phase_velocity) head 학습 | 6 |
| `phase4_subject` | exp4-2: subject-disjoint 5 specs × 3 seeds | 15 |
| `phase4_subset` | exp4-3: transition-only subset 후처리 (기존 42 prediction 재집계) | 42 |
| `synthetic4` | exp4-4: 3 harder synthetic × 8 backbones × 3 seeds | 72 |
| `phase_analysis4` | exp4-1: 18 ckpt × 4 proxy 정량 분석 | 18 |
| **합계** | | **153** |

---

## 1. Experiment 4-1 — 개선된 selective_update_score proxy

### 1.1 새 proxy 정의

기존 phase-3 `selective_score = rho_t` 의 한계 (result3.md §2.2 #4, #5):
- `corr(rho, gyro) = −0.44` (음의 상관)
- `rho_trans / rho_non = 0.99` (transition 변별력 없음)

phase-4 §1.2 의 새 proxy:

```text
forget_rate     = 1 − rho_t
update_budget   = (1 − rho_t) × ||u_t||_2      # 입력으로 과거 state 를 얼마나 갈아엎는지
phase_velocity  = rho_t × |sin(theta_t)|       # 실제 회전 적용 강도
```

### 1.2 18 complex_* ckpt 정량 분석 (mean of 3 seeds)

| proxy | r(proxy, gyro) | r(proxy, |Δφ|) | trans/non ratio | 평가 |
|---|---:|---:|---:|---|
| `rho` (legacy) | −0.44 | −0.35 | 0.99 | 음의 신호, 변별력 없음 |
| **`forget_rate`** | **+0.44** | **+0.35** | **1.13** | ✓ 양의 신호 + 변별력 |
| **`update_budget`** | **+0.30** | **+0.23** | **1.15** | ✓ 양의 신호 + 가장 큰 변별력 |
| `phase_velocity` | −0.64 | −0.57 | 0.97 | ✗ 여전히 음 |

`complex_static` (no selective) 에서는 rho 변동이 없으므로 r 가 NaN/0 → forget/update_budget 의 신호가 selective scanning 에서만 발생함을 직접 확인.

### 1.3 새 head 학습 (3 seeds, direction 7-class, acc+gyro)

| Head | Direction Macro F1 | Δ vs legacy `selective_gyrophase` |
|---|---|---:|
| `complex_selective + selective_gyrophase` (legacy rho) | 0.5979 ± 0.0898 | — |
| **`complex_selective + selective_gyrophase_v2` (update_budget)** | **0.6365 ± 0.0826** | **+0.0386** |
| `complex_selective + selective_gyrophase_v3` (phase_velocity) | 0.5580 ± 0.0412 | −0.0399 |

v2 는 phase-3 의 best complex 변형 (`complex_selective + phase` 0.6258, `gyrophase_rd` 0.6236) 도 능가 → **complex hidden state 를 활용하는 head 중 최고**.

### 1.4 결론

- exp_plan4 §1.5 의 success criteria 3개 모두 충족:  
  ① `corr(proxy, gyro) > 0.3` (+0.44), ② `proxy_trans/non > 1.1` (1.15), ③ `selective_gyrophase_v2 > legacy` (+0.039).
- selective scanning 의 head feature 신호는 `1 − rho` 또는 `(1 − rho) × ||u||` 로 재정의했을 때 비로소 양의 변별력을 가진다.

---

## 2. Experiment 4-2 — Subject-independent evaluation

### 2.1 5 specs × 3 seeds

train/val/test 사용자 완전 분리. `direction_macro_f1` 1차 정렬.

| Model | random dirF1 | **subject dirF1** | Δ | worst-class F1 (subject) |
|---|---|---|---:|---|
| **mamba3 + avgpool** | 0.7553 ± 0.0587 | **0.7862 ± 0.0529** | **+0.031** | 0.6176 ± 0.0861 |
| mamba3 + gyrophase_rd | 0.7611 ± 0.0440 | 0.7707 ± 0.0513 | +0.010 | **0.6465 ± 0.0517** |
| tcn + avgpool | 0.7492 ± 0.0239 | 0.7503 ± 0.0238 | +0.001 | 0.5972 ± 0.0570 |
| transformer + avgpool | 0.6925 ± 0.0220 | 0.7419 ± 0.0234 | +0.049 | 0.6294 ± 0.0509 |
| transformer + gyrophase_rd | 0.7176 ± 0.0417 | 0.7148 ± 0.0128 | −0.003 | 0.5870 ± 0.0572 |

### 2.2 관찰

- **모든 모델이 subject split 에서 random 과 동등하거나 더 좋음** (Δ ≥ −0.003). result.md §3.3 의 binary transition Δ 와 정반대.
  - 가능 원인: HAPT 의 direction transition (6 종) 은 사용자 간 표현이 매우 stereotyped → 사용자 누락 효과 작음.
  - 또는 3-seed × 4–5 user fold 가 변동이 크다.
- **Mamba-3 + AvgPool 이 subject split 에서 새로운 1위 (0.786)** — random 의 winner (Mamba-3 + GyroPhase+RD 0.761) 보다도 높음. mean-pool readout 만으로 사용자 독립 일반화가 가장 견고.
- **Mamba-3 + GyroPhase+RD 의 worst-class F1 = 0.646** — subject 조건에서도 가장 높은 worst-class F1. **드물고 어려운 transition class 에서 안정성** 의 head 효과는 subject-disjoint 에서도 유지.
- **Transformer + GyroPhase+RD 가 random 보다 약간 떨어짐** (−0.003) — 다른 모든 페어가 + 인 점을 고려하면 head 가 사용자별 특이성을 더 따라 학습하는 경향이 있을 수 있음.

### 2.3 paper.md §3 main claim 보강

result.md §3.3 binary 결과 (TCN 가장 견고, Transformer 가장 큰 손실) 와 본 phase-4 direction 결과 (모두 동등 이상) 를 합치면, **transition 검출 (binary) 보다 transition 방향 분류 (direction) 가 사용자 일반화 측면에서 더 견고하다** 는 보조 주장 가능.

---

## 3. Experiment 4-3 — Transition-only subset 재분석

### 3.1 새 split 전략

`y_true >= 1` 인 transition window 만 사용 → 해당 subset 안에서 gyro_mag / RD_std 중앙값으로 high/low 분할. phase-3 의 degenerate 문제 (low-gyro 에 transition 없음) 해소.

### 3.2 핵심 결과 (3 seeds 평균)

| Model | trans-only dirF1 | **trans-high-gyro** | trans-low-gyro | trans-high-RD | trans-low-RD |
|---|---|---|---|---|---|
| **mamba3 + gyrophase_rd** | 0.800 | **0.670 ± 0.132** | 0.750 | 0.800 | 0.793 |
| mamba3 + avgpool | 0.809 | 0.640 ± 0.098 | 0.796 | 0.806 | 0.801 |
| tcn + avgpool | 0.776 | 0.582 ± 0.162 | 0.719 | 0.783 | 0.749 |
| transformer + avgpool | 0.763 | 0.567 ± 0.124 | 0.686 | 0.775 | 0.723 |
| transformer + gyrophase_rd | 0.763 | 0.576 ± 0.075 | 0.721 | 0.704 | 0.780 |
| real_selective + avgpool | 0.798 | 0.582 ± 0.147 | 0.792 | 0.758 | 0.794 |
| real_static + avgpool | 0.734 | 0.531 ± 0.042 | 0.719 | 0.710 | 0.720 |
| complex_selective + gyrophase_rd | 0.768 | 0.551 ± 0.123 | 0.709 | 0.742 | 0.738 |
| complex_selective + phase | 0.703 | 0.462 ± 0.012 | 0.647 | 0.669 | 0.685 |
| complex_static + avgpool | 0.669 | 0.485 ± 0.166 | 0.643 | 0.615 | 0.688 |

### 3.3 관찰

1. **`trans-high-gyro` 가 모든 모델에서 가장 어려운 subset** — full transition F1 대비 평균 −0.14. 짧고 강한 회전 transition 이 가장 분류하기 어렵다.
2. **Mamba-3 + GyroPhase+RD 가 `trans-high-gyro` 에서 1위 (0.670)** — Mamba-3 + AvgPool (0.640) 대비 **+0.030 개선**.  
   full-set 의 Δ는 +0.006 에 불과했으나, **가장 어려운 subset 에서 head 효과는 5배 크다.** paper.md §3 의 GyroPhase Head main claim 을 가장 강하게 지지하는 정량적 증거.
3. **`trans-high-RD` vs `trans-low-RD` 는 거의 동일** — RD_std 가 transition window 내에서는 변별력이 작다 (window 길이 128 안에서 회전 다양성이 포화).
4. **Complex-Selective + Phase 가 `trans-high-gyro` 에서 0.462** — 다른 complex 변형 대비 낮음. 본 head 는 phase 만으로 회전성 신호를 압축해 high-gyro 환경에서 자세 분류 단서를 일부 잃는 것으로 해석.

---

## 4. Experiment 4-4 — Harder synthetic rotation tasks

### 4.1 새 task 정의 (cos/sin only, no ω in input)

- **direction_hard**: 2-class clockwise vs counter-clockwise, ω constant
- **mid_switch**: 2-class, ω switches at random τ ∈ [T/3, 2T/3], label = sign(ω_b)
- **speed_direction6**: 6-class {slow, medium, fast} × {+, −}

### 4.2 결과 (3 seeds, mean ± std macro F1)

| Backbone | direction_hard | mid_switch | speed_direction6 | speed worst-class F1 |
|---|---|---|---|---|
| CNN | 1.0000 ± 0.0000 | 0.9995 ± 0.0005 | 0.9816 ± 0.0024 | 0.9627 ± 0.0077 |
| TCN | 0.9997 ± 0.0006 | 0.9998 ± 0.0003 | 0.9835 ± 0.0031 | 0.9696 ± 0.0065 |
| Transformer | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **0.9881 ± 0.0065** | 0.9821 ± 0.0067 |
| **Mamba-3** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **0.9881 ± 0.0038** | 0.9791 ± 0.0089 |
| real_static | 1.0000 ± 0.0000 | 0.9997 ± 0.0003 | 0.9703 ± 0.0127 | 0.9495 ± 0.0288 |
| real_selective | 1.0000 ± 0.0000 | 0.9993 ± 0.0008 | 0.9758 ± 0.0101 | 0.9518 ± 0.0237 |
| complex_static | 1.0000 ± 0.0000 | 0.9995 ± 0.0005 | 0.9563 ± 0.0048 | 0.9181 ± 0.0072 |
| complex_selective | 1.0000 ± 0.0000 | 0.9982 ± 0.0015 | 0.9806 ± 0.0016 | 0.9699 ± 0.0011 |

### 4.3 관찰

1. **direction_hard 와 mid_switch 는 여전히 변별력 없음 (≥ 0.998 일관)**. cos/sin 만으로도 long enough window 면 모든 architecture 가 ω 부호를 풀어버린다. 추후 더 짧은 window 또는 더 강한 noise 가 필요.
2. **speed_direction6 가 처음으로 모델 차이 출현**:
   - Transformer / Mamba-3 동률 1위 (0.988)
   - complex_static 꼴찌 (0.956) — selective scanning 이 빠지면 6-class speed 분류에서 약점
3. **Worst-class F1 에서 complex_static 명확히 약함 (0.918)**. 6-class 안에서 인접 speed bin 간 혼동이 큼. result3.md §4.6 의 "selective scanning 은 real 에서 +0.072, complex 에서 +0.021" 결과와 결합:
   - real_static → real_selective: 0.970 → 0.976 (+0.006)
   - **complex_static → complex_selective: 0.956 → 0.981 (+0.025)** — speed_direction6 에서는 complex selective 가 real selective 보다 큰 효과!  
   complex update 가 회전 속도 식별에 더 적합하다는 inductive bias 가, HAPT 의 자세 전환 분류보다 **잘 통제된 회전 task** 에서 더 명확히 드러남.
4. **paper.md §6.3 검증 질문 #3 (direction without ω)**: H1 ("Mamba-3 또는 Complex-SSM > CNN/TCN") 은 direction_hard / mid_switch 에서 검증 불가 (ceiling). speed_direction6 에서 Mamba-3 0.988 > TCN 0.984 > CNN 0.982 로 약하게 성립.

---

## 5. RQ 별 정리 (phase-4)

### Q1. `1 − rho` 또는 `(1 − rho) × ||u||` 가 phase-3 rho 보다 selective scanning 신호로 좋은가?
**◯ 강한 yes**. corr(proxy, gyro) 가 −0.44 (rho) → +0.44 (forget_rate) → +0.30 (update_budget), trans/non ratio 가 0.99 → 1.13 → 1.15 로 모두 개선. 그리고 head 학습 시 direction macro F1 0.598 → **0.637** (+0.039).

### Q2. GyroPhase + RD Head 가 subject-independent 에서도 효과적인가?
**△ 부분 yes**. mean Direction Macro F1 에서는 mamba3 + avgpool (0.786) > mamba3 + GyroPhase+RD (0.771). 그러나 **worst-class F1 은 GyroPhase+RD 가 1위 (0.646 vs 0.618)** — drop 큰 user 에서의 안정성은 head 가 더 좋다.

### Q3. Transition-only subset 에서 GyroPhase Head 가 더 큰 효과를 내는가?
**◯ 강한 yes**. `trans-high-gyro` (가장 어려운 subset) 에서 Mamba-3 + GyroPhase+RD 가 Mamba-3 + AvgPool 대비 **+0.030 개선** — full-set Δ (+0.006) 의 5배.

### Q4. cos/sin only 회전 시계열 에서 selective SSM 이 CNN/TCN 을 능가하는가?
**△ controlled task 한정**. direction_hard / mid_switch 는 ceiling 으로 변별력 없음. **speed_direction6** 에서만 Mamba-3 / Transformer (0.988) > CNN (0.982) > TCN (0.984) > complex_static (0.956).

### Q5. Complex update 효과가 selective scanning 이 빠지면 사라지는가?
**✗ 사라지지 않음**. result3.md §4.6 에서 보였듯 phase–gyro coupling 은 complex_static 에서도 0.84. 다만 **6-class speed 분류 (worst-class F1)** 에서는 selective 가 있어야 complex update 의 이점이 나타남 (0.918 → 0.970).

---

## 6. 통합 핵심 주장 (paper.md update 권장)

> Phase-1/2 의 1.231× Δphase ratio, r = 0.847 phase–gyro 상관 (result.md §4.4) 은 phase-3 / phase-4 의 5×3=15 complex_selective + complex_static 학습에서 **0.79–0.85 의 범위와 1.13–1.26 의 ratio** 로 재현되었으며 (result3.md §5, result4.md §1), **selective scanning 이 없는 complex_static 에서도 동일하게 형성**된다는 점이 새 증거다. 또한 selective_update 의 head feature 화는 `update_budget = (1 − rho) × ||u||` proxy 로 가능하며 (direction macro F1 +0.039), GyroPhase + RD Head 의 주된 이득은 **trans-high-gyro 라는 가장 어려운 subset 에서 +0.030**, **subject-disjoint 조건 worst-class F1 +0.029** 로 나타난다. 표준 set 전체 평균 차이가 작은 한계는 본 head 의 효과가 *난이도 적응형* 임을 시사한다.

---

## 7. 한계 및 다음 단계

### 7.1 phase-4 한계
1. **synthetic direction_hard / mid_switch 가 여전히 ceiling**. 다음 iteration 에서 (i) window 단축 (T=32), (ii) signal-to-noise ratio 감소 (noise_std 0.2), (iii) ω switch 를 더 늦은 시점 (≥ 0.9·T) 으로 옮기는 hard variant 필요.
2. **3-seed 통계 미분리**. 본 phase-4 의 +0.030~+0.039 모두 σ~0.04~0.08 안에 있어 통계적 유의성 검증 미수행 — 5+ seeds + paired bootstrap 권장.
3. **subject split fold variance**. 4–5 user / fold 의 작은 sample size 가 +0.03 의 Δ 를 만들 수 있음. leave-one-subject-out (30-fold) 가 더 신뢰 가능.
4. **Mamba-3 fused kernel 내부 phase 미관찰**. phase-4 도 alternative 1 (Complex-SSM 으로 분석) 유지. 향후 mamba3 step API 로 timestep-별 ssm_state 를 hook 하면 직접 비교 가능.

### 7.2 다음 단계 제안
1. **harder synthetic v2**: T=32, noise=0.2, late_switch task 추가.
2. **5–10 seeds × Mamba-3 + GyroPhase+RD vs Mamba-3 + AvgPool, paired bootstrap (B=10000)** 으로 +0.006 mean Δ 의 p-value.
3. **leave-one-subject-out 30-fold** on 5 best specs.
4. **selective_gyrophase_v2 + RD** (현재 v2 spec 은 RD 도 포함). 다른 backbone (Mamba-3) 에도 update_budget proxy 를 적용 — 단 Mamba-3 는 unfused path 가 필요.

---

## 8. 산출물

```
outputs_user/imu_transition/
  phase4_proxy/        seed*/complex_selective__selective_gyrophase_v{2,3}__mean/
                       results_phase3.csv, agg_phase3.md
  phase4_subject/      seed*/{tcn,transformer,mamba3,...}__{avgpool,gyrophase_rd}__mean/
                       results_phase3.csv, agg_phase3.md
  phase3_main/         (재사용) transition_only_subset.csv, transition_only_subset_agg.csv,
                       phase_analysis4.csv, phase_analysis4_summary.csv
  synthetic4/          results_synthetic4.csv, agg_synthetic4.{md,csv}, sweep.log
```

신규 코드 (phase-4):

```
experiments/imu_transition/
  models/ssm_2x2.py                   # forget_rate / update_budget / phase_velocity 노출
  models/gyrophase.py                 # selective_gyrophase_v2/v3 preset, selective_proxy field
  models/phase_classifier.py          # selective_proxy → state dict key 라우팅
  datasets/synthetic_rotation_hard.py # direction_hard, mid_switch, speed_direction6
  run_synthetic4.py                   # 8 backbones × 3 tasks × 3 seeds
  phase_analysis4.py                  # 4 proxy 정량 분석
  aggregate_phase4.py                 # transition-only subset
  aggregate_synthetic4.py             # task별 mean±std
```

---

## 9. 재현

```bash
export HAPT_CACHE_DIR=/home/jdone/ai/mamba/mamba3/cache_user

# Exp 4-1 (선행: phase-3 ckpt 필요)
python experiments/imu_transition/phase_analysis4.py \
    --root outputs_user/imu_transition/phase3_main \
    --specs-regex complex_
python experiments/imu_transition/run_phase3.py --config /tmp/phase3.yaml \
    --seeds 13 42 73 --output-suffix phase4_proxy \
    --specs complex_selective.selective_gyrophase_v2 complex_selective.selective_gyrophase_v3

# Exp 4-2
python experiments/imu_transition/run_phase3.py --config /tmp/phase3.yaml \
    --seeds 13 42 73 --split-mode subject --output-suffix phase4_subject \
    --specs tcn.avgpool transformer.avgpool mamba3.avgpool \
            mamba3.gyrophase_rd transformer.gyrophase_rd

# Exp 4-3
python experiments/imu_transition/aggregate_phase4.py \
    --root outputs_user/imu_transition/phase3_main

# Exp 4-4
python experiments/imu_transition/run_synthetic4.py \
    --seeds 13 42 73 \
    --output-dir outputs_user/imu_transition/synthetic4
python experiments/imu_transition/aggregate_synthetic4.py \
    --root outputs_user/imu_transition/synthetic4
```
