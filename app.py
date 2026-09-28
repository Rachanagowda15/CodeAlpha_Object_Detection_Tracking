import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Load YOLO model
model = YOLO("yolo11n.pt")

# Create Deep SORT tracker
tracker = DeepSort(
    max_age=30,
    n_init=2,
    nms_max_overlap=1.0
)

# Choose input source
print("\n==============================")
print(" CodeAlpha Object Detection")
print("==============================")
print("1. Webcam")
print("2. Recorded Video")

choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
    source_name = "Webcam"

elif choice == "2":
    cap = cv2.VideoCapture("videos/test_video.mp4")
    source_name = "Recorded Video"

else:
    print("Invalid choice.")
    exit()

if not cap.isOpened():
    print(f"Error: Could not open {source_name}.")
    exit()

print(f"{source_name} started successfully.")
print("Press Q to quit.")
print("Object Detection and Tracking Started")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # YOLO object detection
    results = model(frame, verbose=False)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Ignore low-confidence detections
            if confidence < 0.5:
                continue

            # Deep SORT format:
            # ([x, y, width, height], confidence, class_name)

            width = x2 - x1
            height = y2 - y1

            detections.append(
                ([x1, y1, width, height], confidence, class_name)
            )

    # Update Deep SORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Get bounding box
        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        # Get detected class
        class_name = track.get_det_class()

        if class_name is None:
            class_name = "Object"

        # Label
        label = f"{class_name} ID: {track_id}"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw label
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display output
    cv2.imshow(
        "CodeAlpha - Object Detection and Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()