
import numpy as np
import matplotlib.pyplot as plt


def calc_intrinsics(r):
    assert len(r.cameras) == 1
    camera = r.cameras[1]

    K = np.zeros([3,3])

    fx, fy, cx, cy = get_camera_params(camera)

    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy

    K[2,2] = 1
    return K


def get_camera_params(camera):
    if camera.model_name == 'SIMPLE_RADIAL':
        f, cx, cy, k = camera.params
        fx = f
        fy = f
    elif camera.model_name == 'OPENCV':
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
    else:
        print('Unknown camera format')
        input()
        return

    return fx, fy, cx, cy


def world2screen(point3d, R, t, r, with_radial_dist=False, k_mult=1):

    # f, cx, cy, k = r.cameras[1].params

    # k = k * k_mult

    K = calc_intrinsics(r)

    if len(point3d) == 3:
        X = np.concatenate([point3d, np.ones([1])])
    else:
        X = point3d

    Rt = np.concatenate([R, t[:, None]], axis=-1)
    # print(Rt.dtype, K.dtype, X.dtype)
    X_cam = Rt @ X
    # print(X_cam, 'here1', X_cam.dtype)
    # equal to image.transform_to_image(X[:-1])

    X_hom = X_cam / X_cam[-1]


    if False:
        # disable radial distortion, not compatible with simply transforming line
        # works only if sampling N points in 3D and transforming them via distortion

        camera = r.cameras[1]

        if camera.model_name == 'SIMPLE_RADIAL':
            _, _, _, k1 = camera.params
        elif camera.model_name == 'OPENCV':
            _, _, _, _, k1, k2, _, _ = camera.params


        x, y = X_hom[0], X_hom[1]

        radial_dist = ((x) ** 2 + (y) ** 2) ** (1/2)
        if camera.model_name == 'SIMPLE_RADIAL':
            delta_x = x * k1 * (radial_dist ** 2)
            delta_y = y * k1 * (radial_dist ** 2)
        elif camera.model_name == 'OPENCV':
            delta_x = x * k1 * (radial_dist ** 2) + x * k1 * (radial_dist ** 4)
            delta_y = y * k1 * (radial_dist ** 2) + y * k1 * (radial_dist ** 4)

        x_hat = x + delta_x
        y_hat = y + delta_y

        X_hom_hat = np.array([x_hat, y_hat, 1])

        X_hom = X_hom_hat

    return (K @ X_hom)[:2]


def sample_line(p, v):
    linspace = np.linspace(-10, 10, 100)
    return p + np.asarray([v * z for z in linspace])


import pylineclip as lc
def clip_line(imhw, line_on_screen):
    p0x = line_on_screen[0][1]
    p0y = line_on_screen[0][0]
    p1x = line_on_screen[1][1]
    p1y = line_on_screen[1][0]
    out = lc.cohensutherland(xmin=0, xmax=imhw[0]-1, ymin=0, ymax=imhw[1]-1, x1=p0x, y1=p0y, x2=p1x, y2=p1y)

    return np.array(out)


def fit_line(points):
    # compute SVD of the centered points
    mean = points.mean(axis=0)
    centered_points = points - mean
    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)

    # first principal component
    dir = Vt[0]

    return mean, dir


import imageio
def write_mp4(name, src, fps=10):

    if type(src) == str:

        src = os.path.normpath(src)
        if src[-1] != '*':
            src = src + '/*'
        src = sorted(glob(src))

    if type(src[0]) == str:
        src = [plt.imread(fpath) for fpath in src]

    imageio.mimwrite(name + ".mp4", src, "mp4", fps=fps)


### visualisation, utils

from PIL import Image
import io


def figsize(im, scale=10):
    return [x / max(list(im.shape)) * scale for x in im.shape[:2]][::-1]


def fig2im(f, show=False, with_alpha=False, is_notebook=False):

    # f: figure from previous plot (generated with plt.figure())
    buf = io.BytesIO()
    buf.seek(0)
    if is_notebook:
        plt.savefig(buf, format='png', bbox_inches='tight',transparent=True, pad_inches=0)
    else:
        plt.savefig(buf, format="jpg")
    if not show:
        plt.close(f)
    im = Image.open(buf)
    # return without alpha channel (contains only 255 values)
    return np.array(im)[..., : 3 + with_alpha]