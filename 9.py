import cv2
import numpy as np

def find_circles(image_path, min_radius, max_radius, roundness_threshold, area_threshold):
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store candidate circles
    circles = []

    # Loop over contours
    for contour in contours:
        # Approximate contour as a polygon
        approx = cv2.approxPolyDP(contour, epsilon=0.01*cv2.arcLength(contour, True), closed=True)

        # Calculate area and perimeter of contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Ignore contours with area below threshold
        if area < area_threshold:
            continue

        # Fit a circle to the contour and calculate roundness
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius**2
        roundness = 4 * np.pi * area / perimeter**2

        # Ignore circles with radius below threshold
        if radius < min_radius:
            continue

        # Ignore circles with radius above threshold
        if radius > max_radius:
            continue

        # Ignore circles with low roundness
        if roundness < roundness_threshold:
            continue

        # Ignore circles with aspect ratio too far from 1
        aspect_ratio = max(radius, 1) / max(perimeter, 1)
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            continue

        # Add circle to list of candidates
        circles.append((contour, approx, roundness, circle_area))

    # Sort candidate circles by roundness (most round first)
    circles.sort(key=lambda c: c[2], reverse=True)

    # Find best circle among candidates
    best_circle = None
    max_similarity = 0

    for i, (contour, approx, roundness, circle_area) in enumerate(circles):
        # Calculate similarity to ideal circle
        similarity = circle_area / (np.pi * (max_radius ** 2))

        # If similarity is too low, break out of loop
        if similarity < 0.5:
            break

        # Update best circle if this circle has higher similarity
        if similarity > max_similarity:
            best_circle = contour
            max_similarity = similarity

    # Draw best circle on image
    if best_circle is not None:
        cv2.drawContours(image, [best_circle], -1, (0, 0, 255), 2)

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    circles = find_circles(frame, 0.5, 2, 0.75, 100)
    if len(circles) > 0:
        # 在图像中显示检测到的圆
        for circle in circles:
            x, y, r = circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()