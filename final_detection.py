import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import cvzone
import math
import random
from ultralytics import YOLO
from tkinter import scrolledtext

model = YOLO(r"D:\bone_detection\best (3).pt")


classNames = ['angle','fracture','line,messed_up_angle']
myColor = (0, 0, 255)

precautions_set = {
    'angle': [
        """Wear Protective Gear: When engaging in activities with a risk of hand injury, such as sports or construction work, always wear appropriate protective gear like gloves and wrist guards.

        Avoid Repetitive Strain: If your work or hobbies involve repetitive hand movements, take frequent breaks to reduce the risk of overuse injuries.

        Exercise and Strengthen: Regularly perform hand and wrist exercises to strengthen the muscles and improve flexibility. This can help prevent fractures.

        Use Proper Technique: When lifting heavy objects or performing tasks that involve force, use proper lifting techniques and body mechanics to reduce strain on your hands and wrists.

        Be Cautious on Slippery Surfaces: Be cautious when walking on slippery surfaces, especially during wet or icy conditions. Use handrails and maintain balance to prevent falls.""",
    ],
    'fracture': [
        """Proper Fist Techniques: If you're involved in martial arts or self-defense training, ensure you're using proper fist techniques to minimize the risk of hand fractures during strikes.

        Ergonomic Workstation: If you work at a computer, maintain an ergonomic workstation with an appropriately positioned keyboard and mouse. This can reduce the risk of repetitive stress injuries in the hands and wrists.

        Avoid Hand Fatigue: Be mindful of hand fatigue during tasks that require prolonged gripping or fine motor skills. Take breaks and stretch your hands and fingers.

        Warm-Up Exercises: Before engaging in physically demanding activities or sports, perform warm-up exercises specifically targeting your hands and wrists to prepare them for action.

        Wrist Support: Consider wearing wrist supports or braces if you have a history of hand or wrist injuries. These can provide added stability and protection.""",
    ],
    'line': [
        """Check Sports Equipment: If you play sports, regularly inspect and maintain your sports equipment, such as gloves, to ensure they are in good condition and can provide proper protection.

        Avoid Excessive Force: Be cautious when using hand tools or machinery. Avoid excessive force, and make sure you're using the correct tools for the task.

        Proper Handwashing: Practicing good hand hygiene, including proper handwashing techniques, can help prevent infections that might lead to complications and hand injuries.

        Safety in the Kitchen: Use caution in the kitchen to avoid cuts and burns. Always handle knives and hot cookware with care.

        Child Safety: Teach children about hand safety, including the proper handling of sharp objects and staying away from potentially dangerous machinery.""",
    ],
    'messed_up_angle': [
        """Regular Check-ups: If you have a medical condition that affects your bones or joints, or if you've previously experienced hand injuries, consider regular check-ups with a healthcare professional to monitor your hand health.

        Nutrition: Maintain a balanced diet rich in calcium and other essential nutrients that support bone health. Strong bones are less prone to fractures.

        Hand Therapy and Rehabilitation: Seek the guidance of a hand therapist or occupational therapist who specializes in hand injuries. They can design a personalized rehabilitation program to improve hand strength, flexibility, and function during recovery.

        Splint and Support: Follow your doctor's advice on wearing a splint or brace to immobilize the fractured hand. Wearing it consistently and as recommended can help promote proper healing.

        Active Finger Exercises: While your hand is in a splint or cast, perform gentle finger exercises daily to maintain finger mobility. These exercises may include making circles with each finger or tapping your fingertips together.""",
    ],
}

def process_image():
    global image_label

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if file_path:
        img = cv2.imread(file_path)
        results = model(img)

        precaution_text = ""
        fractures_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])

                if conf > 0.5:
                    class_precautions = precautions_set.get(classNames[cls], [])
                    precaution_text += "\n".join(class_precautions)

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

                    fractures_detected = True

        if not fractures_detected:
            precaution_text = "No fractures detected! Keep smiling. 😃"

        processed_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        processed_image = ImageTk.PhotoImage(processed_image)

        image_label.config(image=processed_image)
        image_label.image = processed_image

        precautions_text.delete(1.0, tk.END)  # Clear the existing text
        precautions_text.insert(tk.END, precaution_text)

root = tk.Tk()
root.title("Hand Fracture Prevention ")

window_width = 1024
window_height = 768
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

process_button = tk.Button(root, text="Upload and Detect", command=process_image, bg="#007acc", fg="#ffffff", font=("Arial", 14), padx=20, pady=10)
process_button.pack(pady=20)

content_frame = tk.Frame(root, bg="#ffffff")
content_frame.pack(expand=True, fill=tk.BOTH)

fracture_frame = tk.Frame(content_frame, bg="#f2f2f2", padx=20, pady=20)
fracture_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fracture_label = tk.Label(fracture_frame, text="Detected Fractures:", font=("Arial", 18, "bold"), fg="#333333", bg="#f2f2f2")
fracture_label.pack()

fractures_label = tk.Label(fracture_frame, text="", font=("Arial", 16), fg="#333333", bg="#f2f2f2")
fractures_label.pack()

precautions_frame = tk.Frame(content_frame, bg="#ffffff", padx=20, pady=20)
precautions_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

precautions_title = tk.Label(precautions_frame, text="Guidelines to Follow:", font=("Arial", 18, "bold"), fg="#333333", bg="#ffffff")
precautions_title.pack()

precautions_text = scrolledtext.ScrolledText(precautions_frame, width=50, height=10, font=("Arial", 14))
precautions_text.pack(fill=tk.BOTH, expand=True)

image_label = tk.Label(fracture_frame)
image_label.pack()
root.mainloop()