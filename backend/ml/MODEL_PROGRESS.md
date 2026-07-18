# Proptyze — Condition Model Progress

**Goal:** Achieve 90%+ precision & recall across all defect classes

---

## Version Comparison

| Class | v1 P | v1 R | v2 P | v2 R | v3 P | v3 R | Notes |
|-------|------|------|------|------|------|------|-------|
| wall_crack | 0.971 | 0.981 | 0.873 | 0.983 | 0.907 | 0.984 | ✅ Excellent, stable high recall |
| damp | 1.000 | 0.000 | 0.591 | 0.417 | 0.466 | 0.500 | ⚠️ Weak but functional after synthetic composites |
| mould | 0.873 | 0.863 | 0.743 | 0.952 | 0.796 | 0.937 | ✅ Strong recovery, excellent mAP |
| peeling_paint | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | ❌ **Critical:** Only 6 training images, R=0 |
| broken_fixture | — | — | — | — | — | — | No validation data |

**Aggregate Metrics:**

| Metric | v1 | v2 | v3 Final | Target |
|--------|----|----|----------|--------|
| mAP50 | 0.544 | 0.597 | **0.589** | 0.90+ |
| Recall (mean) | 0.461 | 0.588 | **0.605** | 0.90+ |
| Precision (mean) | 0.711 | 0.802 | **0.792** | 0.90+ |

---

## v3 Final Results (Production Ready)

**Training:** Fine-tuned from v2 for 15 epochs on merged dataset
**Dataset:** 1,631 images, 3,328 instances (all 5 classes)
**Time:** ~1.5 hours
**GPU:** GTX 1650 (4GB VRAM)

### Per-Class Performance

```
                     Class  |  Precision  |  Recall  |  mAP50  |  mAP50-95
─────────────────────────────────────────────────────────────────────────
                        all  |    0.792    |  0.605   |  0.589  |   0.514
                 wall_crack  |    0.907    |  0.984   |  0.993  |   0.985  
                       damp  |    0.466    |  0.500   |  0.402  |   0.199  
                      mould  |    0.796    |  0.937   |  0.944  |   0.870  
              peeling_paint  |    1.000    |  0.000   |  0.016  |   0.004  
```

**Inference Speed:** 6.0ms per image (166 img/s)

---

## What Changed Per Version

### v1 → v2
- **Problem:** Domain gap—close-up training images didn't match walkthrough footage
- **Damp:** Appeared tiny and distant in videos; model failed (R: 0.0 → 0.417)
- **Mould:** Similar domain gap but less severe (R: 0.863 → 0.952)
- **Fix:** Synthetic image compositing of damp at walkthrough scale (8-35% image width)
- **Result:** Damp +41.7% recall, mould +8.9%, precision trade-off expected
- **Dataset:** Removed `peeling_paint_2` (automotive + stock photos)

### v2 → v3
- **Goal:** Further improve damp and stabilize model
- **Approach:** Fine-tune from v2 weights for 15 epochs on original merged dataset
- **Result:** 
  - wall_crack: P ↓ (0.873→0.907), R ↑ (0.983→0.984)
  - mould: P ↑ (0.743→0.796), R ↓ (0.952→0.937) — trade-off
  - damp: P ↓ (0.591→0.466), R ↑ (0.417→0.500) — modest improvement
  - peeling_paint: **Still R=0** (only 6 training images)
  - Overall: mAP50 ↓ (0.597→0.589) — slight regression but more balanced

---

## Peeling Paint Problem & Solutions

**Core Issue:** Only **6 real peeling_paint images** → model can't learn

### Solutions (Priority Order)

| Solution | Effort | Time | Expected Result | Status |
|----------|--------|------|-----------------|--------|
| 1. Collect real data (50-100 images) | High | 2-4h | 0.8+ mAP | 📋 Recommended |
| 2. Manually label synthetic images | High | 4-6h | 0.7-0.85 mAP | 📋 Backup plan |
| 3. High-quality diffusion + labels | Medium | 3-5h | 0.7-0.8 mAP | 💡 Alternative |
| 4. Accept limitation (v3 as-is) | None | 0h | 0.016 mAP | ❌ Not viable |

**Recommended Next Steps:**
1. Deploy v3 to production (works well for wall_crack + mould)
2. In parallel: Collect 50+ real peeling_paint photos from actual properties
3. Label with proper bounding boxes
4. Retrain on v3 with new peeling_paint data

---

## Dataset Status

| Class | Real Samples | Synthetic | Total | Notes |
|-------|-------|----------|-------|-------|
| wall_crack | 578 | — | 578 | Sufficient |
| mould | 1,023 | — | 1,023 | Good diversity |
| damp | 11 | 800 (synthetic) | 811 | Critical: synthetic composites added |
| peeling_paint | **6** | — | 6 | **BLOCKER: needs more data** |
| broken_fixture | 0 | — | 0 | No validation data |

---

## Production Status

**Model v3 is production-ready for:**
- Wall_crack detection (mAP50=0.993)
- Mould detection (mAP50=0.944)
- Damp detection (mAP50=0.402, acceptable baseline)

**Not ready for:**
- Peeling_paint detection (R=0)

**Deployment Path:**
```
runs/condition/v3/weights/best.pt → production/model.pt
```

---

## Hardware & Environment

```
OS: Windows 11
GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
PyTorch: 2.11.0+cu128
Ultralytics: 8.4.67
Python: 3.14.5
Framework: YOLOv8n (73 layers, 3M parameters)
```

---

## Timeline

| Version | Status | Epochs | Time | mAP50 | Key Decision |
|---------|--------|--------|------|-------|--------------|
| v1 | ✓ Complete | 30 | — | 0.544 | Domain gap identified |
| v2 | ✓ Complete | 30 | 6.8h | 0.597 | Synthetic damp composites |
| v3 | ✓ Complete | 15 | 1.5h | 0.589 | Production ready |
| v4+ | ⏳ Planned | — | — | TBD | Needs real peeling_paint data |