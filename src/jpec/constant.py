"""Physical constants and the numerical settings several modules share."""

# physical

# seconds
DAY = 86400.0
# upper bound on a detector's speed v/c, the Earth's orbit plus its rotation
V_MAX = 1.1e-4

# bytes in a giga-byte, as memory is counted
GB = 2.0**30

# numerical: template bank

# fractional bin positions the window's curvature is averaged over
N_CURVATURE_POSITIONS = 200

# numerical: search

# bins a template keeps beyond its arc on each side, for the leakage
ARC_PAD_BINS = 8.0
# orbital phases a template is sampled at, per Nyquist pair: bare Nyquist aliases
THETA_OVERSAMPLE = 4
# slack beyond the Doppler shift and a template's half width
SHIFT_MARGIN_BINS = 16
