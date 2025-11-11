"""
cbDice (centre-line + boundary Dice) in pure TensorFlow.
No EDT required: boundary band is built via soft dilations/erosions.


Requires: tensorflow>=2.9
"""


import tensorflow as tf
from clDice_func import soft_clDice_loss, soft_dice


SMOOTH      = 1e-6
SKEL_ITERS  = 30
DIST_THR    = 2.0   # default boundary half-width (voxels)
BETA        = 1.0   # F_beta emphasis on boundary term (beta>1 favours Tb)
CORE_SHRINK = 1     # GT erosion steps before building skeleton (tolerates ±1 voxel radius changes)


# ──────────────────────────────────────────────────────────── #
# 1) Soft morphology & skeleton
# ──────────────────────────────────────────────────────────── #


def _soft_erode(x):
    if x.shape.rank == 5:   # (B,Z,Y,X,C)
        return -tf.nn.max_pool3d(-x, [1,3,3,3,1], [1,1,1,1,1], "SAME")
    else:                   # (B,H,W,C)
        return -tf.nn.max_pool2d(-x, [1,3,3,1], [1,1,1,1], "SAME")


def _soft_dilate(x):
    if x.shape.rank == 5:
        return  tf.nn.max_pool3d(x, [1,3,3,3,1], [1,1,1,1,1], "SAME")
    else:
        return  tf.nn.max_pool2d(x, [1,3,3,1], [1,1,1,1], "SAME")


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def _ensure_chan(x):
    # add a channel so pooling ops work
    if x.shape.rank in (3,4):
        return x[..., tf.newaxis]
    return x


def _squeeze_chan(x):
    if x.shape.rank in (4,5):
        return tf.squeeze(x, -1)
    return x


def _soft_skel(mask, iters=SKEL_ITERS):
    m = tf.cast(mask, tf.float32)
    if m.shape.rank == 3:      # (H,W) or (Z,Y,X)
        m = m[tf.newaxis, ...]
    if m.shape.rank == 4:      # (B,H,W) or (B,Z,Y,X)
        m = m[..., tf.newaxis] # -> (...,1)
    skel = tf.zeros_like(m)
    img  = m
    for _ in tf.range(iters):
        opened = _soft_open(img)
        delta  = tf.nn.relu(img - opened)
        new    = tf.minimum(skel + tf.nn.relu(delta - skel*delta), 1.0)
        # early stop if no change
        if tf.reduce_sum(new - skel) == 0:
            break
        img  = _soft_erode(img)
        skel = new
    return _squeeze_chan(skel)  # → (B, Z, Y, X) or (B, H, W)


def _erode_binary_steps(y, steps):
    """Binary erosion (hard threshold) applied `steps` times using soft erosion."""
    y = tf.cast(y > 0.5, tf.float32)
    y = _ensure_chan(y)
    for _ in tf.range(tf.cast(steps, tf.int32)):
        y = _soft_erode(y)
    y = tf.cast(y > 0.5, tf.float32)
    return _squeeze_chan(y)


# ──────────────────────────────────────────────────────────── #
# 2) Prepare inputs: pred vs GT handled correctly
# ──────────────────────────────────────────────────────────── #


def _prep_pred(t):
    """Predictions: keep as probabilities; for multi-class logits use union of FG."""
    t = tf.cast(t, tf.float32)
    if t.shape.rank == 5 and t.shape[-1] > 1:
        # softmax over classes, union of all foreground (exclude background ch 0)
        t = tf.nn.softmax(t, axis=-1)[..., 1:]
        t = tf.reduce_max(t, axis=-1)    # (B, …)
    elif t.shape.rank == 5 and t.shape[-1] == 1:
        t = tf.squeeze(t, -1)
    return tf.clip_by_value(t, 0.0, 1.0) # rank=4 or 3


def _prep_true(t):
    """Labels: binary; for one-hot labels take union of foreground channels."""
    t = tf.cast(t, tf.float32)
    if t.shape.rank == 5 and t.shape[-1] > 1:
        t = tf.reduce_max(tf.cast(t[..., 1:] > 0.5, tf.float32), axis=-1)
    elif t.shape.rank == 5 and t.shape[-1] == 1:
        t = tf.squeeze(t, -1)
    else:
        t = tf.cast(t > 0.5, tf.float32)
    return t


# ──────────────────────────────────────────────────────────── #
# 3) Symmetric-ish boundary band and cbDice (F_beta)
# ──────────────────────────────────────────────────────────── #


def _symmetric_band_from_label(y_l, k_band):
    """
    y_l: (B,Z,Y,X) or (B,H,W) binary label.
    k_band: tf.int32 half-width in voxels.
    Returns B with a band (~k voxels) around the true boundary.
    """
    y = tf.cast(y_l > 0.5, tf.float32)
    y = tf.stop_gradient(y)
    y = _ensure_chan(y)  # -> (...,1)


    erode = y
    dilate = y
    k_band = tf.cast(k_band, tf.int32)


    # iterate k times with soft morph. ops
    for _ in tf.range(k_band):
        erode  = _soft_erode(erode)
        dilate = _soft_dilate(dilate)


    erode_b  = tf.cast(erode  > 0.5, tf.float32)
    dilate_b = tf.cast(dilate > 0.5, tf.float32)
    band = tf.clip_by_value(dilate_b - erode_b, 0., 1.)
    return _squeeze_chan(band)


def _boundary_dice_in_band(y_true, y_pred, k_band):
    y_p = _prep_pred(y_pred)
    y_l = _prep_true(y_true)
    B   = _symmetric_band_from_label(y_l, k_band)


    # sum over spatial dims
    axes = list(range(1, y_p.shape.rank))
    num = tf.reduce_sum(B * y_p * y_l, axis=axes)
    den = tf.reduce_sum(B * y_p,       axis=axes) + tf.reduce_sum(B * y_l, axis=axes)
    return (2.0 * num + SMOOTH) / (den + SMOOTH)  # per-sample


def _Ts_gt_core_skeleton_recall(y_true, y_pred, core_shrink=CORE_SHRINK):
    """
    Topology term: recall of a 'core' GT skeleton inside the predicted mask.
    The core is obtained by eroding the GT by `core_shrink` voxels before skeletonisation.
    This makes Ts insensitive to small (e.g. ±1 voxel) radius changes.
    """
    y_l = _prep_true(y_true)      # binary GT
    y_p = _prep_pred(y_pred)      # soft prediction


    if core_shrink > 0:
        y_core = _erode_binary_steps(y_l, core_shrink)
    else:
        y_core = y_l


    S_core = _soft_skel(y_core)   # skeleton of eroded GT


    axes = list(range(1, S_core.shape.rank))
    num = tf.reduce_sum(S_core * y_p, axis=axes)
    den = tf.reduce_sum(S_core,       axis=axes)
    return (num + SMOOTH) / (den + SMOOTH)


@tf.function
def cbdice_loss(y_true, y_pred, dist_thr=DIST_THR, beta=BETA):
    """
    y_true, y_pred: (B,Z,Y,X,1) or (B,H,W,1) or multi-class logits.
    dist_thr: boundary half-width (voxels). We'll round and clamp to ≥2.
    beta: F_beta emphasis on boundary term (beta>1 favours Tb).
    """
    # symmetric boundary Dice restricted to a k-voxel collar
    k_band = tf.maximum(tf.cast(1, tf.int32), tf.cast(tf.round(dist_thr), tf.int32))
    Tb = _boundary_dice_in_band(y_true, y_pred, k_band)


    # topology term: core GT skeleton recall inside prediction
    Ts = _Ts_gt_core_skeleton_recall(y_true, y_pred, CORE_SHRINK)


    # F_beta-style harmonic mean
    b2 = tf.cast(beta * beta, tf.float32)
    cb = ((1.0 + b2) * Ts * Tb + SMOOTH) / (b2 * Ts + Tb + SMOOTH)


    return tf.reduce_mean(1.0 - cb)


def soft_dice_cbdice_loss(alpha=0.1, beta=BETA):
    """
    Combined Dice + cbDice loss.  alpha weights the cbDice term.
    """
    def loss(y_true, y_pred, dist_thr=DIST_THR):
        dice = soft_dice(y_true, y_pred)                    # 1 – Dice coeff
        cb   = cbdice_loss(y_true, y_pred, dist_thr, beta)  # this IS a loss
        return (1.0 - alpha)*dice + alpha*cb
    return loss

def topo_boundary_loss(alpha: float,
                      dist_thr: float,
                      beta: float = BETA):

    def loss(y_true, y_pred):
        dice_l = soft_dice(y_true, y_pred)              # 1 - Dice
        cl_l   = soft_clDice_loss(y_true, y_pred)             # 1 - clDice
        cb_l   = cbdice_loss(y_true, y_pred,
                             dist_thr=dist_thr,
                             beta=beta)                 # 1 - cbDice

        topo_l = (1.0 - alpha) * cl_l + alpha * cb_l
        return 0.5 * (dice_l + topo_l)

    return loss



# ──────────────────────────────────────────────────────────── #
# 4) (Optional) quick test helpers, as you had before...
# ──────────────────────────────────────────────────────────── #


def dice_coeff(a, b):
    i = tf.reduce_sum(a*b, [1,2,3])
    d = tf.reduce_sum(a, [1,2,3]) + tf.reduce_sum(b, [1,2,3])
    return (2*i + SMOOTH)/(d + SMOOTH)


def make_cylinder_5d(r=4, g=64):
    """Return cylinder as (1, Z, Y, X, 1)."""
    zz, yy, xx = tf.meshgrid(tf.range(g), tf.range(g), tf.range(g), indexing='ij')
    mask = tf.cast(((yy - g//2)**2 + (xx - g//2)**2) <= r*r, tf.float32)
    return mask[tf.newaxis, ..., tf.newaxis]


def eval_radius_choices_for_r(r_true, delta_r_list, grid=64,
                              dist_thr=DIST_THR, beta=BETA):
    gt = make_cylinder_5d(r_true, g=grid)
    print(f"\n=== True radius r_true = {r_true} vox ===")


    for delta_r in delta_r_list:
        r_pred = max(1, r_true + delta_r)
        pred   = make_cylinder_5d(r_pred, g=grid)


        d  = dice_coeff(gt, pred)[0].numpy().item()
        cl = 1.0 - soft_clDice_loss(gt, pred).numpy().item()
        cb = 1.0 - cbdice_loss(gt, pred, dist_thr=dist_thr, beta=beta).numpy().item()


        label = "GT radius"
        if delta_r < 0:
            label = f"Smaller (r={r_pred})"
        elif delta_r > 0:
            label = f"Larger  (r={r_pred})"


        print(f"{label:18s} | Δr={delta_r:+d} "
              f"| Dice={d:6.4f}  clDice={cl:6.4f}  cbDice={cb:6.4f}")


if __name__ == "__main__":
    true_radii   = [2, 3, 4, 6, 8]
    delta_r_list = [-1, 0, +1]


    print("Pairwise radius preference test for cbDice")
    print(f"(DIST_THR={DIST_THR}, BETA={BETA}, CORE_SHRINK={CORE_SHRINK})")


    for r_true in true_radii:
        eval_radius_choices_for_r(r_true, delta_r_list,
                                  grid=64,
                                  dist_thr=DIST_THR,
                                  beta=BETA)
