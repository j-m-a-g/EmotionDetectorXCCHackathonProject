from tkinter import *
from tkinter import ttk

from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

from threading import Thread

# Initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id = "face-emotion-recog/1", # Roboflow model
    video_reference = 0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction = render_boxes
)

mainWindow = Tk()
mainWindow.title("Emotion Detector")
mainWindow.geometry("300x130")

howManySecondsToStayLabel = ttk.Label(mainWindow, text = " Once you click, \"Launch\", in order to exit the current\n session, you must close the app's running Python\n console or terminal window. Alternatively, you can\n attempt to resize the \"Emotion Detector\" window to\n intentionally crash it as well.")
howManySecondsToStayLabel.pack(side = TOP, pady = 5)

# Function to simultaneously start and join the webcam video stream
def pipeline_start_join():
    # Create a separate thread for the video stream pipeline to execute on
    pipelineThread = Thread(target=pipeline.start())
    pipelineThread.start()

launchButton = ttk.Button(mainWindow, text = "Launch Session", command = pipeline_start_join)
launchButton.pack(side = TOP)

mainWindow.mainloop()
