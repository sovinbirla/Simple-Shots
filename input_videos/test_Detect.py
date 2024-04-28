import cv2
import numpy as np


def detect_court_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    court_lines = []

    for cnt in contours:
        if cv2.arcLength(cnt, True) > 100:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 2:
                x1, y1 = approx[0][0]
                x2, y2 = approx[1][0]
                court_lines.append([(x1, y1), (x2, y2)])

    return court_lines


def homography_animation(court_lines, image):
    h, w = image.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(
        [[100, 100], [w - 100, 100], [w - 100, h - 100], [100, h - 100]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    legend_size = (600, 400)
    legend = np.zeros((legend_size[1], legend_size[0], 3), dtype=np.uint8)

    for i in range(30):
        alpha = i / 30
        beta = 1 - alpha
        legend_lines = []

        for line in court_lines:
            p1 = (
                int(alpha * line[0][0] + beta * line[1][0]),
                int(alpha * line[0][1] + beta * line[1][1]),
            )
            p2 = (
                int(beta * line[0][0] + alpha * line[1][0]),
                int(beta * line[0][1] + alpha * line[1][1]),
            )
            legend_lines.append([p1, p2])

        for line in legend_lines:
            p1 = (
                int(line[0][0] * legend_size[0] / w),
                int(line[0][1] * legend_size[1] / h),
            )
            p2 = (
                int(line[1][0] * legend_size[0] / w),
                int(line[1][1] * legend_size[1] / h),
            )
            cv2.line(legend, p1, p2, (255, 255, 255), 2)

        cv2.imshow("Court Legend", legend)
        cv2.waitKey(30)
        legend = np.zeros((legend_size[1], legend_size[0], 3), dtype=np.uint8)

    cv2.destroyAllWindows()


# Load the image
input_image = cv2.imread("test.png")

# Detect court lines
court_lines = detect_court_lines(input_image)

# Perform homography animation
homography_animation(court_lines, input_image)
