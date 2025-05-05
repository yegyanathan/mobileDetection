import os
import cv2
import time
import smtplib

from email import encoders
from ultralytics import YOLO
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


# ========== CONFIG ==========
MODEL_PATH = "yolo12x.pt"  # Use custom model if trained on phones
VIDEO_SOURCE = 0  # Change to path for video file or CCTV URL
COOLDOWN_SECONDS = 30  # Email cooldown

SENDER_EMAIL = "<EMAIl ID>"
SENDER_PASSWORD = "<APP CODE>"
RECEIVER_EMAIL = "<EMAIl ID>"
# ============================

# Load model
model = YOLO(MODEL_PATH)

# Video
cap = cv2.VideoCapture(VIDEO_SOURCE)
last_alert_time = 0

def send_email_with_image(image_path):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = "⚠️ Mobile Phone Detected"

    msg.attach(MIMEText("A mobile phone was detected. See attached image.", "plain"))

    with open(image_path, "rb") as f:
        mime = MIMEBase("application", "octet-stream")
        mime.set_payload(f.read())
        encoders.encode_base64(mime)
        mime.add_header("Content-Disposition", f"attachment; filename=" + os.path.basename(image_path))
        msg.attach(mime)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("📧 Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]
    detections = results.boxes

    phone_detected = False

    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls].lower() == "cell phone" and conf > 0.25:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Email alert logic
    current_time = time.time()
    if phone_detected and (current_time - last_alert_time) > COOLDOWN_SECONDS:
        snapshot_path = f"snapshot_{int(current_time)}.jpg"
        cv2.imwrite(snapshot_path, frame)
        send_email_with_image(snapshot_path)
        last_alert_time = current_time

    # Show for debug (optional)
    cv2.imshow("Phone Monitor", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
