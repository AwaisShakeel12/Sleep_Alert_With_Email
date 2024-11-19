from ultralytics import YOLO
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# Email credentials
password = ""
from_email = ""
to_email = ""

# Initialize SMTP server
server = smtplib.SMTP("smtp.gmail.com:587")
server.starttls()
server.login(from_email, password)


def send_email_with_image(to_email, from_email, object_count, image_path):
    """Sends an email with an attachment image when objects are detected."""
    try:
        message = MIMEMultipart()
        message["From"] = from_email
        message["To"] = to_email
        message["Subject"] = "Security Alert"
        message.attach(MIMEText(f"ALERT - {object_count} objects detected!", "plain"))

        # Attach the image
        with open(image_path, "rb") as image_file:
            mime_base = MIMEBase("application", "octet-stream")
            mime_base.set_payload(image_file.read())
            encoders.encode_base64(mime_base)
            mime_base.add_header("Content-Disposition", f"attachment; filename={image_path}")
            message.attach(mime_base)

        server.sendmail(from_email, to_email, message.as_string())
        print(f"Email sent with attachment: {image_path}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def detect_and_alert(video_path):
    """Detect objects in a video using YOLO and send an email when the label 'sleep' is detected."""
    model = YOLO("downness.pt")  # Load your custom-trained YOLO model
    cap = cv2.VideoCapture(video_path)

    email_sent = False  # Initialize email_sent outside the loop

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        
        # Check if any detection has the label "sleep"
        sleep_detected = False
        for box in results[0].boxes:
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name from the model

            if class_name == "sleep":
                sleep_detected = True
                break  # We only need one "sleep" detection to trigger the alert

        # Send email if 'sleep' is detected and no email has been sent yet
        if sleep_detected and not email_sent:
            image_path = "detected_sleep.jpg"
            cv2.imwrite(image_path, frame)  # Save the frame with the detection
            send_email_with_image(to_email, from_email, len(results[0].boxes), image_path)
            email_sent = True  # Mark email as sent to avoid duplicates
        elif not sleep_detected:
            email_sent = False  # Reset email_sent when no "sleep" is detected

        # Show the YOLO-processed frame
        annotated_frame = results[0].plot()  # YOLO draws boxes and labels automatically
        cv2.imshow("YOLO Object Detection", annotated_frame)

        # Press 'q' to stop the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video stream stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()


# Start detection
video_path = r"C:\Users\Awais Shakeel\Desktop\Untitled video - Made with Clipchamp (10).mp4"
detect_and_alert(video_path)
