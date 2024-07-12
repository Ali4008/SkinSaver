import sys
import openai
import cv2
import torch
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QTabWidget
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# Set your OpenAI API key
openai.api_key = OPENAI-API

class ChatBotApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("SkinSaver")
        self.setGeometry(100, 100, 800, 600)


        # Set up the central widget
        central_widget = QTabWidget()
        self.setCentralWidget(central_widget)

        # Chatbot tab
        self.chatbot_tab = QWidget()
        chatbot_layout = QVBoxLayout(self.chatbot_tab)

        # Set up the title label
        title_label = QLabel("SkinSaver")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333333;")
        chatbot_layout.addWidget(title_label)

        # Set up the chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 12))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #F5F5F5;
                color: #000000;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        chatbot_layout.addWidget(self.chat_display)

        # Set up the input layout
        input_layout = QHBoxLayout()

        # Set up the input field
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Arial", 12))
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        input_layout.addWidget(self.input_field)

        # Set up the send button
        send_button = QPushButton("Send")
        send_button.setFont(QFont("Arial", 12, QFont.Bold))
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        input_layout.addWidget(send_button)

        chatbot_layout.addLayout(input_layout)

        # Connect the send button click to the send_message method
        send_button.clicked.connect(self.send_message)

        # Also connect pressing the Enter key to the send_message method
        self.input_field.returnPressed.connect(self.send_message)

        # Initialize the messages list
        self.messages = [{"role": "system", "content": "You are a knowledgeable dermatologist assistant. Provide helpful and accurate recommendations for skin problems detected."}]

        # Camera tab
        self.camera_tab = QWidget()
        camera_layout = QVBoxLayout(self.camera_tab)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #CCCCCC; border-radius: 5px;")
        camera_layout.addWidget(self.video_label)

        # Add a label for additional features or information
        info_label = QLabel("Skin Condition Detection")
        info_label.setFont(QFont("Arial", 16, QFont.Bold))
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #333333; margin-top: 10px;")
        camera_layout.addWidget(info_label)

        # Add tabs to the central widget
        central_widget.addTab(self.chatbot_tab, "Chatbot")
        central_widget.addTab(self.camera_tab, "Camera")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        import pathlib
        from pathlib import Path
        pathlib.PosixPath = pathlib.WindowsPath

        # Load the YOLO model
        model_path = r'C:\Users\Hp\Desktop\RT Tech\Yolo_Skin_Conditions\kaggle\working\best_model.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message:
            return

        # Display user message
        self.chat_display.append(f"<b style='color: blue'>User:</b> {user_message}")

        # Add user message to the conversation history
        self.messages.append({"role": "user", "content": user_message})

        # Call OpenAI API to get a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )

        # Extract the assistant's reply
        reply = response["choices"][0]["message"]["content"]

        # Display assistant's reply
        self.chat_display.append(f"<b style='color: green'>Assistant:</b> {reply}\n")

        # Add assistant's reply to the conversation history
        self.messages.append({"role": "assistant", "content": reply})

        # Clear the input field
        self.input_field.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Perform skin problem detection
            results = self.model(frame)

            # Draw the results on the frame
            for *xyxy, conf, cls in results.xyxy[0].tolist():
                label = f'{results.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatBotApp()
    window.show()
    sys.exit(app.exec_())
