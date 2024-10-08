from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id = "face-emotion-recog/1", # Roboflow model
    video_reference = 0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction = render_boxes
)
pipeline.start()
pipeline.join()
