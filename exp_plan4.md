# exp_plan4: Phase-3 후속 — selective proxy, subject-indep, transition-only subset, harder synthetic

`todo.md` 4개 항목을 정량 실험으로 풀어내는 phase-4 계획. Phase-3 (`exp_plan3.md`, `result3.md`) 의 4가지 한계 (§7.1) 중 1, 2, 5, 7 항목을 다룬다.

| Phase-3 한계 (result3.md §7.1) | 본 phase-4 실험 |
|---|---|
| #2 subject-independent eval 미수행 | Experiment 2 |
| #3 synthetic rotation task 미수행 / direction trivial | Experiment 4 |
| #5 RD subset의 degeneracy | Experiment 3 |
| `corr(sel, gyro) < 0`, `sel_ratio ≈ 1.00` → selective_score 의 head 신호 약함 | Experiment 1 |

---

## Experiment 1. selective_update_score proxy 개선

### 1.1 문제

Phase-3 의 selective_score 는 `rho_t = sigmoid(rho_proj(x))` (= retention coefficient). 분석 결과:

- `corr(sel, gyro) = −0.12 ~ −0.62` (음의 상관)
- `sel_trans / sel_non = 0.99` (transition 구간에서도 거의 동일)

이는 selective scanning 의 정성적 동작 (입력이 클 때 과거 state 망각) 과 일관되나, classifier head feature 로서의 신호는 약하다 (`selective_gyrophase` head 가 plain `gyrophase` 보다 떨어짐).

### 1.2 새 proxy 후보

`update_strength_t = (1 − rho_t) * ||u_t||`

- "현재 step 에서 새 입력이 과거 state 를 얼마나 대체하는가" 의 budget 해석.
- `1 − rho` 는 transition-driven update strength 의 직접 측정값.
- `||u_t||` 은 입력 projection magnitude — 입력 정보량.

추가 후보:

```text
proxy_v1 = (1 − rho_t) * ||u_t||              # update budget
proxy_v2 = (1 − rho_t)                         # forget rate alone
proxy_v3 = rho_t * |Δphase_t| / |φ_t|          # phase-velocity proxy
```

### 1.3 분석 방법

`ComplexSelectiveSSMBlock` 의 `expose_hidden=True` 출력에 다음 추가:

```python
state_dict["update_budget"]  = (1 - rho) * ||(u_real, u_imag)||_2
state_dict["forget_rate"]    = (1 - rho)
state_dict["phase_velocity"] = rho * |Δφ| / (|φ| + 1e-6)
```

기존 18 complex checkpoints (phase-3 sweep) 에 대해 동일한 후처리 스크립트를 새 proxy 로 재실행:

- `corr(proxy, gyro)`
- `corr(proxy, |Δphase|)`
- `mean(proxy)_{trans} / mean(proxy)_{non}`

### 1.4 새 head 변형 추가

Selective GyroPhase Head 의 `selective_score` 입력을 새 proxy 로 교체한 변형 학습 (직접 비교):

- `complex_selective + selective_gyrophase_v2` (update_budget)
- `complex_selective + selective_gyrophase_v3` (phase_velocity)

### 1.5 성공 기준

- `corr(proxy, gyro) > 0` (양의 상관) 이며 `> 0.3`
- `proxy_trans / proxy_non > 1.1`
- `selective_gyrophase_v2` 의 direction macro F1 `> selective_gyrophase` (0.5979)

---

## Experiment 2. Subject-independent evaluation

### 2.1 목적

Phase-3 의 winner 인 `mamba3 + gyrophase_rd` 가 사용자 독립 조건에서도 일반화하는지 확인. result.md §3.3 의 subject-disjoint sweep 과 직접 비교 가능하도록 phase-3 backbone 들을 동일 split 으로 재실험.

### 2.2 모델 (대표)

```text
TCN + AvgPool                      (result.md 의 가장 견고한 baseline)
Transformer + AvgPool              (result.md 의 가장 큰 일반화 손실)
Mamba-3 + AvgPool                  (phase-3 의 새 mean-pool 적용)
Mamba-3 + GyroPhase + RD           (phase-3 winner)
Transformer + GyroPhase + RD       (head portability 확인)
```

5 specs × 3 seeds = 15 runs.

### 2.3 평가

- Direction macro F1, transition F1, worst-class F1
- **Generalization drop**: random_dirF1 − subject_dirF1 (result.md §3.3 와 동일 정의)

### 2.4 성공 기준

```text
Mamba-3 + GyroPhase+RD 의 subject-indep Δ 가 Transformer 보다 작음
TCN 은 result.md (Δ = 0) 와 유사하게 견고할 것으로 예상
```

---

## Experiment 3. Transition-only subset 재분석

### 3.1 문제

Phase-3 의 RD subset 분석 (result3.md §4.4) 은 high/low gyro 가 거의 transition/non-transition split 과 일치 → low-gyro 칼럼이 0.0 으로 degenerate.

### 3.2 새 split 전략

**Transition window 만으로** subset 분석:

```text
1. y_true ∈ {1..6} 인 window 만 선택
2. 해당 window 의 gyro magnitude / RD_std 분포의 중앙값으로 high/low 분할
3. 각 half 에서 7-class direction macro F1 (class 1..6) 계산
```

### 3.3 측정 metric

| Subset | Metric |
|---|---|
| trans-high-gyro | direction macro F1 (classes 1..6) |
| trans-low-gyro | direction macro F1 |
| trans-high-RD | direction macro F1 |
| trans-low-RD | direction macro F1 |
| trans-high-gyro ∩ high-RD | direction macro F1 |

### 3.4 비교 모델

Phase-3 의 14 specs 모두 — 추가 학습 없이 `test_predictions.json` 후처리만으로 가능.

### 3.5 성공 기준

```text
Mamba-3 + GyroPhase+RD 의 trans-high-RD F1 이 Mamba-3 + AvgPool 대비 +ε
Complex-Selective + Phase 의 trans-high-gyro F1 이 Complex-Selective + AvgPool 대비 +ε
```

---

## Experiment 4. Harder synthetic rotation task

### 4.1 문제

result.md §4.3 의 synthetic direction task 는 입력에 `ω_t` 가 포함되어 모든 모델이 100% 정확도 → 변별력 없음.

### 4.2 새 task 설계

#### Task A. Direction without ω (paper.md §6.3 검증 #3)

```text
입력: [cos θ_t, sin θ_t]                       (2 channels)
라벨: 회전 방향 ∈ {clockwise, counter-clockwise}
```

`ω` 를 적분된 phase 변화로부터 모델이 추정해야 함.

#### Task B. Direction switches mid-window

```text
입력: [cos θ_t, sin θ_t]
ω_t = ω_a (t < T/2), ω_b (t ≥ T/2),  ω_a, ω_b independent random
라벨: t = T 에서의 회전 방향 (=sign(ω_b))
```

선택적 상태 갱신이 강해야 풀리는 task.

#### Task C. Speed-class classification

```text
입력: [cos θ_t, sin θ_t]
ω_t = constant, drawn from {slow, medium, fast} × {+, −} → 6 classes
라벨: 6-class speed × direction
```

worst-class F1 분석 가능.

### 4.3 모델

```text
1D-CNN, GRU, TCN, Transformer, Real-Static, Real-Selective,
Complex-Static, Complex-Selective, Mamba-3
```

9 models × 3 tasks × 3 seeds = 81 runs (synthetic 는 빠름, ~1-2 min/run).

### 4.4 평가

| Task | Primary | Secondary |
|---|---|---|
| A | macro F1 | per-class F1 |
| B | macro F1 | first-half-only F1 |
| C | macro F1 | worst-class F1, confusion within slow/medium/fast |

### 4.5 성공 기준 / 가설

```text
H1. Task A: cos/sin only → Mamba-3 또는 Complex-SSM 이 CNN/TCN 보다 우수
H2. Task B: 선택적 상태 갱신 모델이 비선택적보다 우수 (Real-Selective > Real-Static)
H3. Task C: worst-class F1 에서 complex 모델이 real 보다 우수
```

---

## 실행 순서 및 산출물

1. **Experiment 1** (selective proxy):
   - `ssm_2x2.py` 에 새 proxy 노출
   - `phase_analysis3.py` 재실행 (기존 ckpt 사용) → `phase_analysis4.csv`
   - `run_phase3.py --specs complex_selective.selective_gyrophase_v2 ...` 학습 (6 runs)

2. **Experiment 2** (subject-indep):
   - `run_phase3.py --split-mode subject` (15 runs, ~50 min)

3. **Experiment 3** (transition-only subset):
   - `aggregate_phase3.py` 의 `rd_subset_table` 를 transition-only 로 재작성
   - 기존 phase-3 prediction 후처리만 — 추가 학습 없음

4. **Experiment 4** (harder synthetic):
   - `datasets/synthetic_rotation.py` 에 새 task 추가
   - `run_synthetic.py` 확장 (81 runs, ~2 h)

5. **result.md 업데이트**:
   - 본 phase-4 실험을 phase-3 의 §7.1 한계 해결으로 result.md 본문에 직접 통합 (또는 result4.md 신규 작성)
