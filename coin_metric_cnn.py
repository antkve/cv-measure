import numpy as np
from scipy.spatial import distance as dist
import argparse
import imutils
import cv2
from imutils import contours, perspective

TOUCH_POINT_X, TOUCH_POINT_Y = (0, 0)


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_contours(image, blur_amount=7, sensitivity=40, adaptive_threshold=False):
    gsimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gsimage = cv2.GaussianBlur(gsimage, (blur_amount, blur_amount), 0)

    if adaptive_threshold:
        thresh = cv2.adaptiveThreshold(
                gsimage, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 1)
        kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], axis=1)
        edgedetector = cv2.morphologyEx(thresh,
                cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        edgedetector = cv2.Canny(gsimage, sensitivity, 2*sensitivity)
        edgedetector = cv2.dilate(edgedetector, None, iterations=1)
        edgedetector = cv2.erode(edgedetector, None, iterations=1)
        cv2.imshow("mask",edgedetector) 
        cv2.waitKey(0)
        
    dst = cv2.cornerHarris(gsimage,5,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
   
    cnts = cv2.findContours(edgedetector.copy(), cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
            )[1]
    return cnts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True,
            help="path to the input image")
    parser.add_argument("-s", "--reference-size", type=str, required=True,
            help="width of the reference object")
    parser.add_argument("-r", "--reference-threshold", type=float, default=0.6, 
            help="threshold of similarity for recognizing reference object")
    parser.add_argument("-v", "--visualize", action="store_true",
            help="Visualize the process")
    parser.add_argument("-u", "--use-rect", action="store_true",
            help="Look for a rectangular reference object")
    return vars(parser.parse_args())


def center_dist(point, contour):
    x, y = point
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
    cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
    return np.sqrt((x - cx)**2 + (y - cy)**2)


def bounding_box_midlines(box):
    (tl, tr, br, bl) = box
    (topX, topY) = midpoint(tl, tr)
    (bottomX, bottomY) = midpoint(bl, br)
    (leftX, leftY) = midpoint(tl, bl)
    (rightX, rightY) = midpoint(tr, br)
    return ((int(rightX), int(rightY)), (int(leftX), int(leftY))), ((int(topX), int(topY)), (int(bottomX), int(bottomY)))

def bounding_box_dims(box):
    vert_line, horiz_line = bounding_box_midlines(box)
    dH, dW = dist.euclidean(*vert_line), dist.euclidean(*horiz_line)
    print(dH)
    print(dW)
    return dH, dW


def draw_bounding_box(image, box):
    drawing_box = perspective.order_points(box)
    cv2.drawContours(image, [drawing_box.astype("int")], -1, (0, 0, 255), 2)


def bounding_box(contour):
    return np.array(
            cv2.boxPoints(
                cv2.minAreaRect(contour),
                ),
            dtype='int')


def stretch_contour_point(point, vertical, scale_factor):
    x, y = point
    vertical_length = dist.euclidean(*vertical)
    vertical_vec = (vertical[1][0] - vertical[0][0], vertical[1][1] - vertical[0][1])
    top_x, top_y = vertical[0]
    vec_from_top = (x - top_x, y - top_y)
    vertical_projected_length = np.dot(
            vec_from_top, vertical_vec
            ) / dist.euclidean(*vertical)
    vertical_projected_vec = (
            vertical_projected_length * vertical_vec[0] 
                / dist.euclidean(*vertical),
            vertical_projected_length * vertical_vec[1] 
                / dist.euclidean(*vertical))
    new_x, new_y = (x + vertical_projected_vec[0] * (scale_factor - 1), y + vertical_projected_vec[1] * (scale_factor - 1))
    return int(new_x), int(new_y)


def stretch_contour(contour, vert_line, box_height_scale):
    return np.array([np.array(
        [stretch_contour_point((pt[0][0], pt[0][1]), vert_line, box_height_scale)])
            for pt in contour])


def match_square_shape(contour, ref_contour):
    box = bounding_box(contour)
    box_height, box_width = bounding_box_dims(box)
    box_height_scale = box_width / box_height
    vert_line, _ = bounding_box_midlines(box)
    contour = stretch_contour(contour, vert_line, box_height_scale)
    return cv2.matchShapes(
        contour, ref_contour, 1, 0.0)

def line_length(line):
    return dist.euclidean(*line)

def __main__():

    args = parse_args()
    
    image = cv2.imread(args['image'])

    cnts, _ = contours.sort_contours(get_contours(image))
    if not args['use_rect']:
        reference_image = cv2.imread("coin.jpg")
        sample_reference = max(get_contours(reference_image), key=lambda x: cv2.contourArea(x))
    else:
        ref_width, ref_height = args['reference_size'].split('x')
        ref_width, ref_height = float(ref_width), float(ref_height)

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]
    reference = min(
            cnts, key=lambda x: match_square_shape(
                x, sample_reference)
            ) if not args['use_rect'] else min([
                    cnt for cnt in cnts 
                    if len(cv2.approxPolyDP(
                        cnt, (0.1 * cv2.arcLength(cnt, True)), 
                        1, 0.0)
                        ) == 4
                    ], key=lambda x: cv2.contourArea(x))
    print(reference)
    obj = max(cnts, key=lambda x: cv2.contourArea(x))

    boxes = [bounding_box(cnt) for cnt in cnts]
    ref_box = bounding_box(reference)
    ref_height_px, ref_width_px = bounding_box_dims(ref_box)
    pixelsPerCm = ref_width_px / ref_width

    box_dims_px = [bounding_box_dims(box) for box in boxes]
    box_dims_mm = [(box_height / pixelsPerCm, box_width / pixelsPerCm)
            for box_height, box_width in box_dims_px]
    obj_box = bounding_box(obj)
    obj_height_px, obj_width_px = bounding_box_dims(obj_box)
    obj_height_cm, obj_width_cm = (
            obj_height_px / pixelsPerCm, 
            obj_width_px / pixelsPerCm
            )

    ref_poly = cv2.approxPolyDP(reference, 0.1 * cv2.arcLength(reference, True), 1, 0.0)
    ref_box_points = perspective.order_points(np.array([k[0] for k in ref_poly], dtype='int'))
    ref_width, ref_height = ref_width * 10, ref_height * 10
    target_box_points = np.array(((ref_box_points[0][0], 200 + ref_box_points[0][1]), (ref_box_points[0][0] + ref_width, 200 + ref_box_points[0][1]), 
        (ref_box_points[0][0] + ref_width, 200 + ref_box_points[0][1] + ref_height), (ref_box_points[0][0], 200 + ref_box_points[0][1] + ref_height)))
    h, result = cv2.findHomography(ref_box_points, target_box_points)

    if args['visualize']:
        orig = image.copy()
        out = cv2.warpPerspective(orig, h, (1000, 900))
        cv2.imshow("coin_metric_cnn.py", out)
        cv2.waitKey(0)
        epsilon = 0.003 * cv2.arcLength(obj, True)
        obj_poly = cv2.approxPolyDP(obj, epsilon, True)
#        cv2.drawContours(orig, [obj_poly], -1, (0, 0, 255), 2)
        cv2.drawContours(orig, [reference], -1, (0, 0, 255), 2)

        box_height, box_width = bounding_box_dims(ref_box)
        box_height_scale = box_width / box_height
        vert_line, _ = bounding_box_midlines(ref_box)
        obj_corrected = stretch_contour(obj, vert_line, box_height_scale)
        obj_corr_height_px, obj_corr_width_px = bounding_box_dims(bounding_box(obj_corrected))
        obj_corr_height_cm, obj_corr_width_cm = obj_corr_height_px / pixelsPerCm, obj_corr_width_px / pixelsPerCm

        
        cv2.putText(
                orig, "FOUND REFERENCE, MEASURING {0:.2f} px/cm".format(
                    pixelsPerCm,
                    ),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
#        cv2.putText(
#                orig, "CERTAINTY: {0:.2f}%".format(
#                    100 * (1 - match_square_shape(reference, sample_reference))
#                    ),
#                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(
                orig, "FOUND SHORTS",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(
                orig, "WIDTH: {0:.2f} cm; HEIGHT: {1:.2f}".format(
                    obj_height_cm,
                    obj_width_cm
                    ),
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        obj_vert, obj_horiz = bounding_box_midlines(obj_box)
        cv2.line(orig, obj_vert[0], obj_vert[1], (0, 255, 0), 2)
        cv2.line(orig, obj_horiz[0], obj_horiz[1], (0, 255, 0), 2)
        cv2.imshow("coin_metric_cnn.py", orig)
        cv2.waitKey(0)


if __name__ == "__main__":
    __main__()
