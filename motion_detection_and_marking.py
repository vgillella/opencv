import cv2

cap = cv2.VideoCapture("opencv/video.mp4")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0

while True:
    ret, current_frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale and find difference
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, current_gray)
    
    # Apply threshold and find contours directly
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 20:
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save frame if motion detected every 10 frames
    if motion_detected and frame_count % 10 == 0:
        cv2.imwrite(f"motion_{frame_count}.jpg", current_frame)
        print(f"Saved motion_{frame_count}.jpg")

    cv2.imshow("Motion Detection", current_frame)
    
    # Update for next iteration
    prev_gray = current_gray
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
