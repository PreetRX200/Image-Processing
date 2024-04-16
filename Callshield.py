import streamlit as st
import cv2
import math
from ultralytics import YOLO
import cvzone
import time
import datetime
import tkinter as tk
from twilio.rest import Client

# Twilio credentials
account_sid = 'ACebecc9f1e84782c17a16a5cbe8a2368d'
auth_token = '84b2a7efaf2233d83ad6719bcd8276db'
client = Client(account_sid, auth_token)

# Set the page config as the first Streamlit command
st.set_page_config(page_title="Call-Sheild: Call Detection App", layout="wide")

def get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def call_detection_app():
    # Sidebar with options
    with st.sidebar:
        st.title("Object Detection App")
        run = st.checkbox("Run Object Detection")
        source_type = st.radio("Source Type", ("Webcam", "RTSP", "HTTP Cam"))
        if source_type == "RTSP":
            source = st.text_input("Enter RTSP URL", "rtsp://example.com/stream")
        elif source_type == "HTTP Cam":
            source = st.text_input("Enter HTTP Cam URL", "http://example.com/stream")
        else:
            source = 0  # Webcam index
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Main content
    alert_msg = st.empty()
    if run:
        cap = cv2.VideoCapture(source)
        model = YOLO('best.pt')
        screen_width, screen_height = get_screen_resolution()  # Use the new function
        classnames = ['phone']
        prev_frame_time = 0
        new_frame_time = 0
        stframe = st.empty()
        call_detected_time = None
        phone_count = 0
        warning_sent = False  # Flag to track if a warning has been sent

        # Create placeholders for buttons
        fps_button_placeholder = st.empty()
        height_button_placeholder = st.empty()
        width_button_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (screen_width, screen_height))
                result = model(frame, stream=True)

                # Initialize a flag to check if a bounding box is found
                bounding_box_found = False
                phone_count_in_frame = 0  # Variable to store the number of phones detected in the current frame

                for info in result:
                    boxes = info.boxes
                    for box in boxes:
                        confidence = box.conf[0]
                        if confidence > confidence_threshold:
                            bounding_box_found = True  # Set the flag to True if a bounding box is found
                            confidence = math.ceil(confidence * 100)
                            Class = int(box.cls[0])
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

                            # Increment phone count for the current frame
                            if Class == 0:
                                phone_count_in_frame += 1

                new_frame_time = cv2.getTickCount()
                fps = 1 / ((new_frame_time - prev_frame_time) / cv2.getTickFrequency())
                prev_frame_time = new_frame_time
                fps = int(fps)

                # Generate unique keys for buttons
                current_time = str(time.time())

                # Update buttons with new values
                fps_button_placeholder.button(f"FPS: {fps}", key=f"fps_{current_time}")
                height_button_placeholder.button(f"Frame Height: {frame.shape[0]}", key=f"height_{current_time}")
                width_button_placeholder.button(f"Frame Width: {frame.shape[1]}", key=f"width_{current_time}")

                stframe.image(frame, channels="BGR")

                # Display an alert if a bounding box is found
                if bounding_box_found:
                    if call_detected_time is None:
                        call_detected_time = time.time()
                        alert_msg.warning(f"Call detected at {datetime.datetime.fromtimestamp(call_detected_time).strftime('%H:%M:%S')}")
                    else:
                        elapsed_time = time.time() - call_detected_time
                        if elapsed_time > 10 and not warning_sent:
                            warning_sent = True
                            message = client.messages.create(
                                body='Warning:It is a no calling Zone. You are requested not to use calling services for your own safety.',
                                from_='+12514511968',
                                to='+919330477877'
                            )
                            st.warning(f'Message sent: {message.sid}')
                        if elapsed_time > 10:
                            alert_msg.error(f"WARNING: NO CALLING ZONE! Number of phones detected: {phone_count_in_frame}")
                        else:
                            alert_msg.warning(f"Call detected for {int(elapsed_time)} seconds. Number of phones detected: {phone_count_in_frame}")

                    # Update the total phone count
                    phone_count += phone_count_in_frame
                else:
                    call_detected_time = None
                    warning_sent = False  # Reset the warning_sent flag
                    alert_msg.info(f"No call detected.")
                    phone_count = 0  # Reset phone_count to 0 when no phone is detected

            else:
                st.warning("Error: Unable to read frame.")
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        alert_msg.info("Check the 'Run Object Detection' checkbox to start.")

def about_page():
    st.title("About")
    st.write("Call-Shield: An calling Object Detection App created by Binary Bits.")
    # Display an image with smaller width
    st.image("sign.jpeg", caption="NO CALLING ZONE", width=350)
    st.write("It uses the power of computer vision and deep learning to detect objects in real-time video streams.")
    st.write("You can use it to keep an eye on people who are bit careless on road, railways, construction site etc!")
    st.write("Feel free to play around with the app and let me know if you have any suggestions or bugs to report.")
    st.write("Happy detecting!")

def main():
    st.sidebar.title("Navigation")
    pages = {
        "Call Detection App": call_detection_app,
        "About": about_page
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()
