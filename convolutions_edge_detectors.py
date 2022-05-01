import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

DERIVATIVE_KERNEL_X = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
DERIVATIVE_KERNEL_Y = DERIVATIVE_KERNEL_X.T
LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])




def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    kernel_size = len(k_size)
    signal_size = len(in_signal)

    padding_size = kernel_size - 1
    padded = np.pad(in_signal, (padding_size, padding_size), 'constant', constant_values=(0, 0))
    num_of_windows = signal_size + kernel_size - 1
    flipped = np.flip(k_size)
    convoloved = np.zeros(num_of_windows)

    for i in range(num_of_windows):
        window = padded[i:i + kernel_size]
        window_dot_product = window.dot(flipped)
        convoloved[i] = window_dot_product

    return convoloved


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    img_height, img_width = in_image.shape
    kernel_height, kernel_width = kernel.shape

    convolved = np.zeros_like(in_image)
    middle_kernel_y, middle_kernel_x = kernel_height // 2, kernel_width // 2
    padding_width, padding_height = kernel_height, kernel_width

    padded = np.pad(in_image, ((padding_height, padding_height), (padding_width, padding_width)),
                    'edge')

    for i in range(img_height):
        for j in range(img_width):
            up_border = i + 1 + middle_kernel_y
            down_border = up_border + kernel_height
            left_border = j + 1 + middle_kernel_x
            right_border = left_border + kernel_width
            patch = padded[up_border: down_border, left_border:right_border]
            convolved[i][j] = (patch * kernel).sum()

    return convolved


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    x_derivative = conv2D(in_image, DERIVATIVE_KERNEL_X)
    y_derivative = conv2D(in_image, DERIVATIVE_KERNEL_Y)
    magnitude = np.sqrt(x_derivative ** 2 + y_derivative ** 2).astype(np.float64)
    directions = np.arctan2(y_derivative , x_derivative).astype(np.float64)
    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian_kernel = generate_gaussian_kernel(k_size)
    return conv2D(in_image, gaussian_kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian_kernel = cv.getGaussianKernel(k_size, 0)
    gaussian_kernel = gaussian_kernel.dot(gaussian_kernel.T)
    return cv.filter2D(in_image, - 1, gaussian_kernel, borderType=cv.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    return edgeDetectionZeroCrossingLOG(img)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    smoothed = blurImage1(img, 31)
    derivative = conv2D(smoothed, LAPLACIAN_KERNEL)
    edge_matrix = np.zeros_like(derivative)
    for i in range(0, derivative.shape[0]):
        for j in range(0, derivative.shape[1]):
            submatrix = derivative[i:i + 3, j:j + 3]
            if is_zero_crossing(submatrix):
                edge_matrix[i][j] = 1
    return edge_matrix


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    height, width = img.shape

    derivative_y = cv.Sobel(img, -1, 0, 1, ksize=3)
    derivative_x = cv.Sobel(img, -1, 1, 0, ksize=3)
    ori = np.arctan2(derivative_y, derivative_x)

    canny_edges = cv.Canny((img * 255).astype(np.uint8), 550, 100)

    radius_range = max_radius - min_radius + 1
    bins = np.zeros((height, width, radius_range))

    Y, X = np.where(canny_edges == 255)

    sins = np.sin(ori[Y, X])
    cosins = np.cos(ori[Y, X])

    radiuses = np.arange(min_radius, max_radius + 1)

    for y, x, sin, cos in zip(Y, X, sins, cosins):
        # vectorized multipication of radiuses and sins cosins
        sin = (radiuses * sin).astype(int)
        cos = (radiuses * cos).astype(int)

        # vectorized addition and substraction
        a1, b1 = x + cos, y + sin
        a2, b2 = x - cos, y - sin

        # plus one for valid bins, using masking a b vectors
        pos_mask = get_mask_for_idxs(a1, b1, height, width)
        bins[b1[pos_mask], a1[pos_mask], pos_mask] += 1
        pos_mask = get_mask_for_idxs(a2, b2, height, width)
        bins[b2[pos_mask], a2[pos_mask], pos_mask] += 1

    # fix it for something more sophisticated if there's time
    y, x, r = np.where(bins > (bins.max()) / 3)

    circles = np.array([x, y, r + min_radius]).T

    return circles


def get_mask_for_idxs(a, b, height, width):
    """
    making masks for a and b, getting the indexes with values between
    0 and width for a, 0 and height for b
    """
    first_mask = np.logical_and(b > 0, a > 0)
    second_mask = np.logical_and(b < height, a < width)
    return np.logical_and(first_mask, second_mask)


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: run image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    k_size = k_size // 2
    padded = cv.copyMakeBorder(in_image, k_size, k_size, k_size, k_size,
                               cv.BORDER_REPLICATE)
    height, width = padded.shape
    ans = np.zeros_like(in_image,dtype=np.float64)

    for y in range(k_size, height - k_size):
        for x in range(k_size, width - k_size):
            pivot_v = padded[y, x]
            neighborhood = padded[y - k_size:y + k_size + 1, x - k_size:x + k_size + 1]

            diff = pivot_v - neighborhood
            diff_gau = np.exp(-0.5 * np.power(diff / sigma_color, 2))

            gaussian_vector = cv.getGaussianKernel(2 * k_size + 1, sigma=sigma_space,ktype=cv.CV_64F)
            gaussian_kernel = gaussian_vector.dot(gaussian_vector.T)

            combo = gaussian_kernel * diff_gau

            convolved_patch = combo * neighborhood

            ans[y - k_size, x - k_size] = convolved_patch.sum() / combo.sum()

    cvs_ans_for_comparison = cv.bilateralFilter(in_image, k_size,
                                                sigma_color, sigma_space,
                                                borderType=cv.BORDER_REPLICATE)
    return cvs_ans_for_comparison, ans



def generate_gaussian_kernel(k_size):
    """
    generate gaussian kernel by conv1D convolution
    :param k_size: odd number
    :return: k*k kernel
    """
    base = np.array([1, 1])
    gaussian_kernel_generator = np.array([1, 1])
    # There is a proof why it is correct.
    for i in range(1, k_size - 1):
        base = conv1D(base, gaussian_kernel_generator)

    base = base.reshape(len(base), -1)
    kernel = base.dot(base.T)
    return normalize(kernel)


def is_zero_crossing(mat):
    max = mat.max()
    min = mat.min()
    return min * max < 0


def normalize(mat):
    return mat / mat.sum()


def is_safe(mat, i, j):
    n, m = mat.shape
    return 0 <= i < n and 0 <= j < m
