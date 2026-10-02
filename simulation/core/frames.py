import numpy as np


def normalise_quaternion(q):
    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,).")

    norm = np.linalg.norm(q)

    if norm == 0:
        raise ValueError("Quaternion cannot have zero magnitude.")

    return q / norm


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quaternion_to_rotation_matrix(q):
    q = normalise_quaternion(q)

    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def body_to_inertial(vector_body, quaternion):
    rotation = quaternion_to_rotation_matrix(quaternion)
    return rotation @ np.asarray(vector_body, dtype=float)


def inertial_to_body(vector_inertial, quaternion):
    rotation = quaternion_to_rotation_matrix(quaternion)
    return rotation.T @ np.asarray(vector_inertial, dtype=float)
