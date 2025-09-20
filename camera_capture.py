import cv2

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Save every 30th frame
    if frame_count % 30 == 0:
        cv2.imwrite(f"frame_{frame_count}.jpg", frame)
        print(f"Saved: frame_{frame_count}.jpg")

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()