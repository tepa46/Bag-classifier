import cv2
import numpy as np


def collect_image_features(image):
    """
    Collects a set of all features values for the given image.

    Parameters:
        image: The input image for which features will be computed.

    Returns:
        list: A list containing features values computed for the given image.
    """

    statistics = [
        get_hypothesis_1_feature_value(image),
        get_hypothesis_2_feature_value(image),
        get_hypothesis_3_feature_value(image),
        get_hypothesis_4_feature_value(image),
        get_hypothesis_5_feature_value(image),
        get_hypothesis_6_feature_value(image),
        get_hypothesis_7_feature_value(image),
        get_hypothesis_8_feature_value(image),
        get_hypothesis_9_feature_value(image),
        get_hypothesis_10_feature_value(image),
    ]

    return statistics


# Мусорные пакеты имеют темные цвета
def get_hypothesis_1_feature_value(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black_mask = gray_image < 20

    return np.count_nonzero(black_mask)


# Пластиковые пакеты и мусорные пакеты часто содержат яркие блики
def get_hypothesis_2_feature_value(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    bright_spots = grad_magnitude > 40

    return np.count_nonzero(bright_spots)


# На изображениях с бумажными пакетами много длинных отрезков
def get_hypothesis_3_feature_value(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=2)

    if lines is not None:
        return len(lines)
    return 0


# Бумажные пакеты имеют более насыщенные цвета
def get_hypothesis_4_feature_value(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_saturation = 191
    mask = (hsv_image[:, :, 1] > lower_saturation)

    return np.count_nonzero(mask)


# Бумажные пакеты часто имеют светлокоричневый цвет
def get_hypothesis_5_feature_value(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([15, 50, 50])
    upper_brown = np.array([45, 255, 200])
    mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    return np.count_nonzero(mask)


# Бумажные пакеты --- матовые
def get_hypothesis_6_feature_value(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var


# Пластиковые пакеты имеют яркие цвета (желтый, синий, белый)
def get_hypothesis_7_feature_value(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([140, 255, 255])

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 50, 255])

    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(image, blue_lower, blue_upper)
    white_mask = cv2.inRange(image, white_lower, white_upper)

    all_color_mask = yellow_mask | blue_mask | white_mask

    count = np.count_nonzero(all_color_mask)

    return count


# Из-за сильно выраженных складок на мусорных пакетах, найденные контуры мусорных пакетов по площади будут меньше,
# чем контуры пластиковых и бумажных пакетов
def get_hypothesis_8_feature_value(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return cv2.contourArea(contours[0])


# Контуры бумажных и пластиковых пакетов имеют меньше углов
def get_hypothesis_9_feature_value(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    approx = cv2.approxPolyDP(contours[0], 0.0005 * cv2.arcLength(contours[0], True), True)

    return len(approx)


# Пластиковые пакеты из-за своей прозрачности могут иметь участки ненасыщенного цвета
def get_hypothesis_10_feature_value(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_saturation = 40

    mask = (hsv_image[:, :, 1] < lower_saturation)

    return np.count_nonzero(mask)
