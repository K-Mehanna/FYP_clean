SPATIAL_MIDPOINT = 0.5

# 2D-calibrated thresholds (pixel-fraction percentiles from BraTS PNG dataset,
# per-class slot volumes across all tumour-containing slices in the 369-patient set)
# Percentiles: p10=0.000608, p25=0.002830, p50=0.008611, p75=0.017179
VOLUME_2D_VERY_SMALL_THRESHOLD = 0.000608   # p10
VOLUME_2D_SMALL_THRESHOLD      = 0.002830   # p25
VOLUME_2D_MEDIUM_V2_THRESHOLD  = 0.008611   # p50
VOLUME_2D_MEDIUM_THRESHOLD     = 0.017179   # p75
