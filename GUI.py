import tkinter as tk
import time
import subprocess
import cv2



# def setBackgroundIMG(root, imgPath):
#     # Set the background image
#     bg_image = tk.PhotoImage(file="C:\\Users\\apoor\\Downloads\\background_1.gif")
#     bg_label = tk.Label(root, image=bg_image)
#     bg_label.place(relx=0.5, rely=0.5, anchor="center")
    # bg_label.image = bg_image

def setBackgroundIMG(root, imgPath):
    # Set the background image
    bg_image = tk.PhotoImage(file=imgPath)
    root.bg_image = bg_image  # Store bg_image as an attribute of root
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(relx=0.5, rely=0.5, anchor="center")


def stop_program(process):
    process.terminate()

def runVM():
    global is_open, process
    # Run the Python code
    path = "virtual_mouse_hands.py"
    if is_open and process != None:
        cur_status.config(text="Virtual Mouse is closing wait few seconds...")
        time.sleep(2)
        try:
            stop_program(process)
            is_open = False
        except:
            cur_status.config(text="Virtual Mouse is still running...")  
    process = subprocess.Popen(["python", path])
    if not is_open: 
        is_open = True
    if(process != None): cur_status.config(text="Virtual Mouse is running...")

def runFAW():
    global is_open, process
    # Run the Python code
    path = "FAW.py"
    if is_open and process != None:
        cur_status.config(text="Finger air writing is closing wait few seconds...")
        time.sleep(2)
        try:
            stop_program(process)
            is_open = False
        except:
            cur_status.config(text="Finger air writing is still running...")     
    process = subprocess.Popen(["python", path])
    if not is_open: 
        is_open = True
    if(process != None): 
        cur_status.config(text="Finger air writing is running...")


root = tk.Tk()

canvas = tk.Canvas(root, width=1280, height=700)
canvas.config(bg="#cfe2f3")       # skin
canvas.pack()

# Call setBackgroundIMG before mainloop
setBackgroundIMG(root, "C:\\Users\\apoor\\Downloads\\background_1.gif")

is_open = False
process = None

# canvas = tk.Canvas(root, width=1280, height=700)
# canvas.config(bg="#cfe2f3")       # skin
# canvas.pack()

# is_open = False
# process = None

button1 = tk.Button(canvas, text="Virtual Mouse", width=15, height=2,command=runVM)
button1.place(relx=0.3, rely=0.5, anchor="center")

button1.config(bg="#1E90FF", fg="white") # light blue

button2 = tk.Button(canvas, text="Finger Air Writing", width=15, height=2,command=runFAW)
button2.place(relx=0.7, rely=0.5, anchor="center")

button2.config(bg="#1E90FF", fg="white") # light blue


cur_status = tk.Label(root, text="Waiting for user input.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
cur_status.pack(side=tk.BOTTOM, fill=tk.X)


root.mainloop()