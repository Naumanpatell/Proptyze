# Proptyze — Condition Model Progress

**Goal:** Achieve 90%+ precision & recall across all defect classes

---

## Version Comparison

| Class | v1 P | v1 R | v2 P | v2 R | v3 P | v3 R | Notes |
|-------|------|------|------|------|------|------|-------|
| wall_crack | 0.971 | 0.981 | 0.771 | 0.993 | — | — | Stable, high recall |
| damp | 1.000 | 0.000 | 0.349 | 0.250 | — | — | Critical blocker; synthetic composites added for v3 |
| mould | 0.873 | 0.863 | 0.738 | 0.956 | — | — | Strong recovery after synthetic compositing |
| peeling_paint | 0.000 | 0.000 | 0.110 | 0.208 | — | — | Dataset cleanup helped; still weak |
| broken_fixture | — | — | — | — | — | — | No validation data |

**Aggregate Metrics:**

| Metric | v1 | v2 | v3 | Target |
|--------|----|----|----|----|
| mAP50 (mean) | 0.544 | 0.613 | — | 0.90+ |
| Recall (mean) | 0.461 | 0.602 | — | 0.90+ |
| Precision (mean) | 0.711 | 0.492 | — | 0.90+ |

---

## What Changed Per Version

### v1 → v2
- **Problem:** Domain gap—close-up training images didn't match walkthrough footage
- **Damp:** Appeared tiny and distant in videos; model failed (R: 0.0)
- **Mould:** Similar domain gap but less severe (R: 0.863)
- **Fix:** SAHI sliced inference + synthetic image compositing
- **Result:** Damp +25% recall, mould +9.2%, but precision dropped (expected trade-off)
- **Dataset:** Removed `peeling_paint_2` (automotive + stock photos)

### v2 → v3
- **Problem:** Damp still weak (P: 0.349, R: 0.250); peeling_paint fragile after cleanup
- **Fix:** Generated 1000 synthetic damp composites at walkthrough scale
  - Damp patches resized to 8–35% of image width
  - Pasted onto clean wall backgrounds (wall_crack, peeling_paint_1)
  - Split 800 train / 200 valid
- **Dataset:** Merged `damp_synthetic` into main `condition/merged`
- **Status:** Training in progress

---

