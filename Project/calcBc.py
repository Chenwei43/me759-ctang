import numpy as np

def getBcSpiral(gx, gy, x,y,z, affine, B0=3):
    """
    concomitant field calc for spiral
    Based on MRM 66:390–401 (2011) DOI 10.1002/mrm.22802
    :param gx: (npts,)
    :param gy: (npts,)
    :param coords: xres by yres by 3 logic coords
    :param affine: 3 by 3 transformation matrix
    :return: xres by yres by pts array
    """

    if affine.ndim == 2:
        affine = affine.flatten()

    f1 = .25 * (affine[0]**2+affine[3]**2)*(affine[6]**2+affine[7]**2) + affine[6]**2 * \
         (affine[1]**2+affine[4]**2) - affine[6] * affine[7] * (affine[0]*affine[1]+affine[3]*affine[4])
    f2 = .25 * (affine[1]**2+affine[4]**2)*(affine[6]**2+affine[7]**2) + affine[7]**2 * \
         (affine[0]**2+affine[3]**2) - affine[6] * affine[7] * (affine[0]*affine[1]+affine[3]*affine[4])
    f3 = .25 * (affine[2]**2+affine[5]**2) * (affine[6]**2+affine[7]**2) \
         + affine[8]**2 * (affine[0]**2+affine[1]**2+affine[3]**2+affine[4]**2) \
         - affine[6] * affine[8] * (affine[0]*affine[2]+affine[3]*affine[5]) \
         - affine[7] * affine[8] * (affine[1]*affine[2]+affine[4]*affine[5])
    f4 = .5 * (affine[1] * affine[2] + affine[4] * affine[5]) * (affine[6]**2-affine[7]**2) \
         + affine[7]*affine[8] * (2*affine[0]**2+affine[1]**2+2*affine[3]**2+affine[4]**2)\
         - affine[6]*affine[7] * (affine[0]*affine[2]+affine[3]*affine[5]) \
         - affine[6]*affine[8] * (affine[0]*affine[1]+affine[3]*affine[4])
    f5 =.5 * (affine[0] * affine[2] + affine[3] * affine[5]) * (affine[7]**2-affine[6]**2) \
        + affine[6] * affine[8] * (2 * affine[0] ** 2 + affine[1] ** 2 + 2 * affine[3] ** 2 + affine[4] ** 2) \
        - affine[6] * affine[7] * (affine[1]*affine[2]+affine[4]*affine[5]) \
        - affine[7] * affine[8] * (affine[0]*affine[1]+affine[3]*affine[4])
    f6 = -.5 * (affine[0]*affine[1] + affine[3]*affine[4]) * (affine[6]**2+affine[7]**2) \
        + affine[6] * affine[7] * (affine[0]**2 + affine[1]**2 + affine[3]**2 + affine[4]**2)


    g_sq = gx**2 + gy**2
    spatial = (f1 * x**2 + f2 * y**2 + f3 * z**2 + f4 * y * z + f5 * x * z + f6 * x * y)
    spatial = np.tile(np.expand_dims(spatial,-1), (1,1,gx.shape[0]))
    B_c = g_sq / (4*B0) * spatial

    return B_c


#def getBc(gx, gy, coords, affine, B0=3):
