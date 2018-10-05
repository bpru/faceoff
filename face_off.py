import cv2
import dlib
import numpy as np

import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# these points are used for aligning 2 faces
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# these points are used for overlay face2 to face1
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + 
                  RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS,]



def get_landmarks(img, detector, predictor):
    """
    get face landmarks for all detected faces
    """
    detected = detector(img, 1)
    print 'detedcted %d faces...'%len(detected)

    landmarks = []

    for i in range(len(detected)):
        points = predictor(img, detected[i]).parts()
        landmark = np.matrix([[point.x, point.y] for point in points])
        landmarks.append(landmark)
    
    return landmarks
    
def transform(points1, points2):
    """
    compute the transformation matrix for points2 to align with points1
    """
    
    # change types for later calculation
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # find the center of both points
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)

    # make the points centered at (0,0)
    points1 -= center1
    points2 -= center2

    # calculate standard deviation of points
    std1 = np.std(points1)
    std2 = np.std(points2)

    # correct the scaling for both points
    points1 /= std1
    points2 /= std2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((std2 / std1) * R,
                     center2.T - (std2 / std1) * R * center1.T)),
                     np.matrix([0., 0., 1.])])

def get_mask(img, landmark):
    res = np.zeros(img.shape[:2], dtype=np.float64)

    for feature in OVERLAY_POINTS:
        points = landmark[feature]
        convex_points = cv2.convexHull(landmark[feature])
        cv2.fillConvexPoly(res, convex_points, color = 1)

    res = np.array([res, res, res]).transpose((1, 2, 0))

    res = cv2.GaussianBlur(res, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return res

def warp_img(img, M, dshape):
    
    output = np.zeros(dshape, dtype=img.dtype)
    cv2.warpAffine(img, M[:2], (dshape[1], dshape[0]), dst=output, 
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output

def color_balance(img1, img2, landmarks1):

    # use the middle point of eyes to do the color balancing
    blur_amount = int(np.linalg.norm(
                  np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                  np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)))
    if blur_amount % 2 == 0:
        blur_amount += 1
    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return (img2.astype(np.float64) * img1_blur.astype(np.float64) /
                                                img2_blur.astype(np.float64))


def label_landmarks(img, landmarks, filename):
    copy = np.copy(img)
    for i in range(len(landmarks)):
        p = tuple(landmarks[i].tolist()[0])
        cv2.putText(copy, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, .2, 255)
    cv2.imwrite(filename, copy)



if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # get the first 2 landmarks, representing the landmarks for the first 2 detected faces
    landmarks1, landmarks2 = get_landmarks(img, detector, predictor)[:2]

    label_landmarks(img, landmarks1, 'face1.jpg')
    label_landmarks(img, landmarks2, 'face2.jpg')

    # compute the transformation matrix for lankmarks2 to align with lankmarks1
    M1 = transform(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    # compute the transformation matrix for lankmarks1 to align with lankmarks2
    M2 = transform(landmarks2[ALIGN_POINTS], landmarks1[ALIGN_POINTS])

    # generate masks
    mask1 = get_mask(img, landmarks2)
    mask2 = get_mask(img, landmarks1)

    cv2.imwrite('mask1.jpg', mask1 * 100)
    cv2.imwrite('mask2.jpg', mask2 * 100)
    
    warped_mask1 = warp_img(mask1, M1, img.shape)
    warped_mask2 = warp_img(mask2, M2, img.shape)

    combined_mask1 = np.max([mask2, warped_mask1], axis=0)
    combined_mask2 = np.max([mask1, warped_mask2], axis=0)

    cv2.imwrite('transformed_mask1.jpg', warped_mask1 * 100)
    cv2.imwrite('transformed_mask2.jpg', warped_mask2 * 100)

    cv2.imwrite('combined_mask1.jpg', combined_mask1 * 100)
    cv2.imwrite('combined_mask2.jpg', combined_mask2 * 100)

    warped_img2 = warp_img(img, M1, img.shape)
    warped_img1 = warp_img(img, M2, img.shape)

    cv2.imwrite('warped_img1.jpg', warped_img1)
    cv2.imwrite('warped_img2.jpg', warped_img2)

    warped_corrected_img2 = color_balance(img, warped_img2, landmarks1)
    warped_corrected_img1 = color_balance(img, warped_img1, landmarks2)

    cv2.imwrite('warped_corrected_img1.jpg', warped_corrected_img1)
    cv2.imwrite('warped_corrected_img2.jpg', warped_corrected_img2)

    output = img * (1.0 - combined_mask1) + warped_corrected_img2 * combined_mask1
    output = output * (1.0 - combined_mask2) + warped_corrected_img1 * combined_mask2

    cv2.imwrite('output.jpg', output)
    print 'finished swapping 2 faces.'