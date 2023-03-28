from typing import Tuple
import numpy as np
from colmap_converter.colmap_utils import Camera as ColmapCamera


def distortion(extra_params, u, v)-> Tuple[float, float]:
    """
    TODO: ChatGPT: translate below into Python

    template <typename T>
    void FOVCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                    T* du, T* dv) {
    const T omega = extra_params[0];

    // Chosen arbitrarily.
    const T kEpsilon = T(1e-4);

    const T radius2 = u * u + v * v;
    const T omega2 = omega * omega;

    T factor;
    if (omega2 < kEpsilon) {
        // Derivation of this case with Matlab:
        // syms radius omega;
        // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
        //                  (radius * omega);
        // simplify(taylor(factor, omega, 'order', 3))
        factor = (omega2 * radius2) / T(3) - omega2 / T(12) + T(1);
    } else if (radius2 < kEpsilon) {
        // Derivation of this case with Matlab:
        // syms radius omega;
        // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
        //                  (radius * omega);
        // simplify(taylor(factor, radius, 'order', 3))
        const T tan_half_omega = ceres::tan(omega / T(2));
        factor = (T(-2) * tan_half_omega *
                (T(4) * radius2 * tan_half_omega * tan_half_omega - T(3))) /
                (T(3) * omega);
    } else {
        const T radius = ceres::sqrt(radius2);
        const T numerator = ceres::atan(radius * T(2) * ceres::tan(omega / T(2)));
        factor = numerator / (radius * omega);
    }

    *du = u * factor;
    *dv = v * factor;
    }
    """
    # omega = extra_params[0]
    # radius2 = u * u + v * v
    # omega2 = omega * omega
    # if omega2 < 1e-4:
    #     factor = (omega2 * radius2) / 3 - omega2 / 12 + 1
    # elif radius2 < 1e-4:
    #     tan_half_omega = tan(omega / 2)
    #     factor = (-2 * tan_half_omega *
    #             (4 * radius2 * tan_half_omega * tan_half_omega - 3)) /
    #             (3 * omega)
    # else:
    #     radius = sqrt(radius2)
    #     numerator = atan(radius * 2 * tan(omega / 2))
    #     factor = numerator / (radius * omega)

    # du = u * factor
    # dv = v * factor

    return 0.0, 0.0


def project_points(pts3d: np.ndarray,
                   w2c: np.ndarray,
                   camera: ColmapCamera):
    """
    Args:
        pts3d: (N, 3)
        w2c: (4, 4)
    
    Returns:
        pts2d: (N, 2)
    """
    width, height = camera.width, camera.height

    pts2d = np.zeros((pts3d.shape[0], 2))
    pts3d_homo = np.vstack((pts3d, np.ones((pts3d.shape[0], 1))))
    ppts3d_c = w2c @ pts3d_homo
    pass
    cen, dir = transform_line(cen, dir, qvec, tvec)
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.params

    cen_uv = cen[:2] / cen[2]
    cen_uv = cen_uv * np.array([fx, fy]) + np.array([cx, cy])
    dir_uv = ((dir + cen)[:2] / (dir + cen)[2]) - (cen[:2] / cen[2])
    dir_uv = dir_uv * np.array([fx, fy])
    dir_uv = dir_uv / np.linalg.norm(dir_uv)
    # Previous wrong implementation:
    # dir_uv = dir[:2] / dir[2]
    # dir_uv = dir_uv * np.array([fx, fy])
    # dir_uv = dir_uv / np.linalg.norm(dir_uv)

    # TODO Assume fx = fy
    # TODO: distort
    thickness = radius / cen[2] * fx
    normal = np.array([-dir_uv[1], dir_uv[0]])
    ub_uv = cen_uv + normal * thickness
    lb_uv = cen_uv - normal * thickness