"""Body-part boxes from pose keypoints, shared by the indexer (parts.py, on Modal) and photo search (on
the Studio). Pure numpy, so either side can import it without the other's dependencies.

Parts: left and right hand (elbow to fingertips), head, torso, legs, feet. Sides are the person's own
(COCO keypoint convention), not the image's.
"""
import numpy as np

KP_CONF = 0.5

(NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP,
 L_KNEE, R_KNEE, L_ANK, R_ANK) = range(17)


def part_boxes(kp: np.ndarray, person: tuple[float, float, float, float]) -> list[tuple[str, str | None, tuple, float]]:
    """(part, side, x1 y1 x2 y2 in frame pixels, confidence) from COCO keypoints (17 x [x, y, conf])."""
    H = person[3] - person[1]
    ok = lambda i, c=KP_CONF: kp[i, 2] >= c
    out = []

    for side, wr, el in (("L", L_WR, L_EL), ("R", R_WR, R_EL)):
        if not ok(wr):
            continue
        w = kp[wr, :2]
        f = w - kp[el, :2]
        n = float(np.hypot(*f))
        if ok(el, 0.3) and n > 1:
            L, d = min(max(n, 0.08 * H), 0.25 * H), f / n
        else:
            L, d = 0.14 * H, np.array([0.0, 1.0])
        # Elbow to fingertips (a hand is ~0.75 forearm lengths long): tattoos, watches and bracelets sit on
        # the forearm and wrist, rings and held objects in the hand.
        c, s = w - d * 0.125 * L, 1.9 * L
        out.append(("hand", side, (c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2), float(kp[wr, 2])))

    shoulders = ok(L_SH, 0.4) and ok(R_SH, 0.4)
    # Shoulder width shrinks in side views; the person's height bounds it from below.
    m = max(float(np.hypot(*(kp[L_SH, :2] - kp[R_SH, :2]))) if shoulders else 0.0, 0.2 * H)

    face = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if ok(i, 0.4)]
    s = 0.9 * m
    if face:
        c, conf = kp[face, :2].mean(0) - [0, 0.15 * s], float(kp[face, 2].mean())   # up a little: hair, hats
    elif shoulders:
        c, conf = (kp[L_SH, :2] + kp[R_SH, :2]) / 2 - [0, 0.75 * m], 0.8 * float(min(kp[L_SH, 2], kp[R_SH, 2]))
    else:
        c = None
    if c is not None:
        out.append(("head", None, (c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2), conf))

    if shoulders:
        xs, ys = [kp[L_SH, 0], kp[R_SH, 0]], [kp[L_SH, 1], kp[R_SH, 1]]
        if ok(L_HIP, 0.3) and ok(R_HIP, 0.3):
            bottom = (kp[L_HIP, 1] + kp[R_HIP, 1]) / 2 + 0.05 * m
            xs += [kp[L_HIP, 0], kp[R_HIP, 0]]
        else:
            bottom = max(ys) + 1.4 * m
        top = min(ys) - 0.1 * m
        if bottom > top + 0.3 * m:
            out.append(("torso", None, (min(xs) - 0.15 * m, top, max(xs) + 0.15 * m, bottom),
                        float(min(kp[L_SH, 2], kp[R_SH, 2]))))

    # Legs (pants, shorts, skirts): hips down to the ankles, or a shin's length below the knees.
    hips = [i for i in (L_HIP, R_HIP) if ok(i, 0.4)]
    knees = [i for i in (L_KNEE, R_KNEE) if ok(i, 0.4)]
    if hips and knees:
        low = [i for i in (L_ANK, R_ANK) if ok(i, 0.4)] or knees
        xs = kp[hips + knees + low, 0]
        bottom = kp[low, 1].max() + (0 if low != knees else 0.25 * H)
        top, pad = kp[hips, 1].min() - 0.03 * H, 0.06 * H
        if bottom > top + 0.15 * H:
            out.append(("legs", None, (xs.min() - pad, top, xs.max() + pad, bottom),
                        float(kp[hips + knees, 2].mean())))

    # Feet (shoes): both ankles, extended down and sideways by about a foot length (~0.15 of height).
    ankles = [i for i in (L_ANK, R_ANK) if ok(i, 0.4)]
    if ankles:
        xs, ys, f = kp[ankles, 0], kp[ankles, 1], 0.13 * H
        out.append(("feet", None, (xs.min() - f, ys.min() - 0.6 * f, xs.max() + f, ys.max() + 0.9 * f),
                    float(kp[ankles, 2].mean())))
    return out
