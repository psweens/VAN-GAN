"""
cbDice (centre-line + boundary Dice) using TF-Addons Euclidean DT.
Requires: tensorflow>=2.9, tensorflow-addons>=0.20.
"""

import tensorflow as tf
import tensorflow_addons as tfa
from scipy.ndimage import distance_transform_edt
from clDice_func import soft_clDice_loss, soft_dice

SMOOTH     = 1e-6
SKEL_ITERS = 30
DIST_THR   = .5                # in voxels

# ──────────────────────────────────────────────────────────── #
# 1) Soft skeleton (unchanged from before)
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
    # correct: erode then dilate
    return _soft_dilate(_soft_erode(x))

def _soft_skel(mask, iters=SKEL_ITERS):
    # ensures shape (...,1) channel so pooling works
    m = tf.cast(mask, tf.float32)
    if m.shape.rank == 3:      # (H,W) or (Z,Y,X)
        m = m[tf.newaxis,...]
    if m.shape.rank == 4:      # (B,H,W) or (B,Z,Y,X)
        m = m[...,tf.newaxis]
    skel = tf.zeros_like(m)
    img  = m
    for _ in tf.range(iters):
        opened = _soft_open(img)
        delta  = tf.nn.relu(img - opened)
        new    = tf.minimum(skel + tf.nn.relu(delta - skel*delta), 1.0)
        if tf.reduce_sum(new - skel) == 0:
            break
        img  = _soft_erode(img)
        skel = new
    return tf.squeeze(skel, -1)  # → (B, Z, Y, X) or (B, H, W)

# ──────────────────────────────────────────────────────────── #
# 2) Prepare input: merge classes & drop channels
# ──────────────────────────────────────────────────────────── #

def _prep(t):
    t = tf.cast(t, tf.float32)
    if t.shape.rank == 5:   # (B, …, C)
        if t.shape[-1] > 1:
            # one-hot → union of all foreground classes
            t = tf.nn.softmax(t, -1)[...,1:]
            t = tf.reduce_max(t, -1)
        else:
            t = tf.squeeze(t, -1)
    return t  # now rank=4 or 3

# ──────────────────────────────────────────────────────────── #
# 3) The cbDice loss – now using SciPy’s EDT instead of TF-Addons
# ──────────────────────────────────────────────────────────── #

@tf.function
def cbdice_loss(y_true, y_pred, dist_thr=DIST_THR):
    """
    y_true, y_pred: shape (B,Z,Y,X,1) or (B,H,W,1) or one-hot logits.
    dist_thr: boundary collar thickness in voxels.
    """
    # 1) flatten classes & channels
    y_p = _prep(y_pred)
    y_l = _prep(y_true)

    # 2) skeletons
    S_p = _soft_skel(y_p)
    S_l = _soft_skel(y_l)

    # 3) inverse‐radius weighting via true Euclidean DT
    #    compute distance‐to‐background at every label‐voxel
    bin_l = tf.cast(y_l > .5, tf.uint8)
    dt_l  = tf.cast(tfa.image.euclidean_dist_transform(bin_l), tf.float32)
    # pick out skeleton voxels
    idx   = tf.where(S_l > .5)
    inv_w = tf.scatter_nd(
        idx,
        1.0 / (tf.gather_nd(dt_l, idx) + SMOOTH),
        tf.shape(y_l, out_type=tf.int64)
    )

    # 4) boundary collar: dist-to-background of the TRUE label
    inv_bg = tf.cast(bin_l < 1, tf.uint8)
    dt_bg  = tf.cast(tfa.image.euclidean_dist_transform(inv_bg), tf.float32)
    B_l    = tf.cast(dt_bg <= dist_thr, y_p.dtype)

    # 5) directional recalls
    Ts = (tf.reduce_sum(inv_w * S_p * y_l, [1,2,3]) + SMOOTH) / \
         (tf.reduce_sum(inv_w * S_p,     [1,2,3]) + SMOOTH)

    Tb = (tf.reduce_sum(B_l * y_p,      [1,2,3]) + SMOOTH) / \
         (tf.reduce_sum(B_l,            [1,2,3]) + SMOOTH)

    cb = (2.0 * Ts * Tb + SMOOTH) / (Ts + Tb + SMOOTH)

    # return mean loss over batch
    return tf.reduce_mean(1.0 - cb)

def soft_dice_cbdice_loss(alpha=0.1):
    """
    Combined Dice + cbDice loss.  alpha weights the cbDice term.
    """
    def loss(y_true, y_pred, dist_thr=DIST_THR):
        dice = soft_dice(y_true, y_pred)             # 1 – Dice coeff
        cb   = cbdice_loss(y_true, y_pred, dist_thr)   # this IS a loss
        return (1.0-alpha)*dice + alpha*cb
    return loss

# ──────────────────────────────────────────────────────────── #
# 4) Quick sanity test on a tiny 3-D cylinder
# ──────────────────────────────────────────────────────────── #

def dice_coeff(a, b):
    i = tf.reduce_sum(a*b, [1,2,3])
    d = tf.reduce_sum(a, [1,2,3]) + tf.reduce_sum(b, [1,2,3])
    return (2*i + SMOOTH)/(d + SMOOTH)

def make_cylinder(r=4, g=32):
    zz, yy, xx = tf.meshgrid(tf.range(g), tf.range(g), tf.range(g), indexing='ij')
    return tf.cast(((yy-g//2)**2 + (xx-g//2)**2) <= r*r, tf.float32)

if __name__ == "__main__":
    grid = 32
    gt    = make_cylinder()[None,...,None]   # (1,Z,Y,X,1)
    p0    = tf.identity(gt)
    p1    = tf.roll(gt, 1, axis=3)           # +1 voxel
    p3    = tf.roll(gt, 3, axis=3)           # +3 voxels

    for name, p in [("Perfect overlap", p0),
                    ("Shift +1 voxel", p1),
                    ("Shift +3 voxels", p3)]:
        d  = dice_coeff(gt, p)[0].numpy().item()
        cl = 1.0 - soft_clDice_loss(gt, p).numpy().item()
        cb = 1.0 - cbdice_loss(gt, p).numpy().item()
        print(f"{name:17s} | Dice={d:.4f}   clDice={cl:.4f}   cbDice={cb:.4f}")