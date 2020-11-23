import numpy as np
import matplotlib.cm
import skimage.io
import skimage.feature
import skimage.filters


def vec2im(V, shape=()):
    '''
    Transform an array V into a specified shape - or if no shape is given assume a square output format.

    Parameters
    ----------

    V : numpy.ndarray
        an array either representing a matrix or vector to be reshaped into an two-dimensional image

    shape : tuple or list
        optional. containing the shape information for the output array if not given, the output is assumed to be square

    Returns
    -------

    W : numpy.ndarray
        with W.shape = shape or W.shape = [np.sqrt(V.size)]*2

    '''

    if len(shape) < 2:
        shape = [np.sqrt(V.size)] * 2
        shape = map(int, shape)
    return np.reshape(V, shape)


def enlarge_image(img, scaling=3):
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.

    Parameters
    ----------

    img : numpy.ndarray
        array of shape [H x W] OR [H x W x D]

    scaling : int
        positive integer value > 0

    Returns
    -------

    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]
        OR
        three-dimensional array of shape [scaling*H x scaling*W x D]
        depending on the dimensionality of the input
    '''

    if scaling < 1 or not isinstance(scaling, int):
        print('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H, W = img.shape

        out = np.zeros((scaling * H, scaling * W))
        for h in range(H):
            fh = scaling * h
            for w in range(W):
                fw = scaling * w
                out[fh:fh + scaling, fw:fw + scaling] = img[h, w]

    elif len(img.shape) == 3:
        H, W, D = img.shape

        out = np.zeros((scaling * H, scaling * W, D))
        for h in range(H):
            fh = scaling * h
            for w in range(W):
                fw = scaling * w
                out[fh:fh + scaling, fw:fw + scaling, :] = img[h, w, :]

    return out


def repaint_corner_pixels(rgbimg, scaling=3):
    '''
    DEPRECATED/OBSOLETE.

    Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    Assumes those image ares will pretty much never show evidence.

    Parameters
    ----------

    rgbimg : numpy.ndarray
        array of shape [H x W x 3]

    scaling : int
        positive integer value > 0

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3]
    '''

    # top left corner.
    rgbimg[0:scaling, 0:scaling, :] = (rgbimg[0, scaling, :] + rgbimg[scaling, 0, :] + rgbimg[scaling, scaling,
                                                                                       :]) / 3.0
    # bottom right corner
    rgbimg[-scaling:, -scaling:, :] = (rgbimg[-1, -1 - scaling, :] + rgbimg[-1 - scaling, -1, :] + rgbimg[-1 - scaling,
                                                                                                   -1 - scaling,
                                                                                                   :]) / 3.0
    return rgbimg


def digit_to_rgb(X, scaling=3, shape=(), cmap='binary'):
    '''
    Takes as input an intensity array and produces a rgb image due to some color map

    Parameters
    ----------

    X : numpy.ndarray
        intensity matrix as array of shape [M x N]

    scaling : int
        optional. positive integer value > 0

    shape: tuple or list of its , length = 2
        optional. if not given, X is reshaped to be square.

    cmap : str
        name of color map of choice. default is 'binary'

    Returns
    -------

    image : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    # create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    image = enlarge_image(vec2im(X, shape), scaling)  # enlarge
    image = cmap(image.flatten())[..., 0:3].reshape([image.shape[0], image.shape[1], 3])  # colorize, reshape

    return image


def hm_to_rgb(R, X=None, scaling=3, shape=(), sigma=2, cmap='bwr', normalize=True):
    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.

    Parameters
    ----------

    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]

    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]

    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.

    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.

    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.

    cmap : str
        optional. color map of choice

    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    # create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    if normalize:
        R = R / np.max(np.abs(R))  # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.) / 2.  # shift/normalize to [0,1] for color mapping

    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[..., 0:3].reshape([R.shape[0], R.shape[1], 3])
    # rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None:  # compute the outline of the input
        # X = enlarge_image(vec2im(X,shape), scaling)
        xdims = X.shape
        Rdims = R.shape

        # if not np.all(xdims == Rdims):
        #     print 'transformed heatmap and data dimension mismatch. data dimensions differ?'
        #     print 'R.shape = ',Rdims, 'X.shape = ', xdims
        #     print 'skipping drawing of outline\n'
        # else:
        #     #edges = skimage.filters.canny(X, sigma=sigma)
        #     edges = skimage.feature.canny(X, sigma=sigma)
        #     edges = np.invert(np.dstack([edges]*3))*1.0
        #     rgb *= edges # set outline pixels to black color

    return rgb


def save_image(rgb_images, path, gap=2):
    '''
    Takes as input a list of rgb images, places them next to each other with a gap and writes out the result.

    Parameters
    ----------

    rgb_images : list , tuple, collection. such stuff
        each item in the collection is expected to be an rgb image of dimensions [H x _ x 3]
        where the width is variable

    path : str
        the output path of the assembled image

    gap : int
        optional. sets the width of a black area of pixels realized as an image shaped [H x gap x 3] in between the input images

    Returns
    -------

    image : numpy.ndarray
        the assembled image as written out to path
    '''

    sz = []
    image = []
    for i in range(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0], gap, sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print('image', i, 'differs in size. unable to perform horizontal alignment')
            print('expected: Hx_xD = {0}x_x{1}'.format(sz[0], sz[1]))
            print('got     : Hx_xD = {0}x_x{1}'.format(rgb_images[i].shape[0], rgb_images[i].shape[1]))
            print('skipping image\n')
        else:
            image = np.hstack((image, gap, rgb_images[i]))

    image *= 255
    image = image.astype(np.uint8)

    print('saving image to ', path)
    skimage.io.imsave(path, image)
    return image
