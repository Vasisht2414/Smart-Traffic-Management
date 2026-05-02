from ultralytics import YOLO
from flask import Flask, render_template, request
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil


flask_app_folder_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

@app.route("/", methods=["GET"])
def starting_page():
    return render_template("index.html")

def import_image(image_path):
    image = Image.open(image_path)
    return image


yolo = YOLO(flask_app_folder_path+"/best.pt") # last.pt

def apply_yolo(yolo_model, image):
    detections_results = yolo_model.predict(image, save=True)
    return detections_results
    

@app.route("/", methods=["POST"])
def predict_image():
    # print("Entered predict_image()")
    if request.method == "POST":
        if "image" not in request.files:
            return "No image part"
        image_file = request.files["image"]
        if image_file.filename == "":
            return "No selected file"

        # Save the uploaded image to a folder (you might need to create the folder)
        uploaded_path = os.path.join(flask_app_folder_path, "static", "uploaded_images", image_file.filename)
        # print("Uploaded Path Before Saving:", uploaded_path)
        image_file.save(uploaded_path)
        
        # Apply YOLO
        image = import_image(uploaded_path)
        detections_results = apply_yolo(yolo, image)
        result_image_path = os.path.abspath(detections_results[0].save_dir) + "\\" + image_file.filename
        print("Result Image Path:", result_image_path)
        # result_image = import_image(result_image_path)
        shutil.copy(result_image_path, os.path.join(flask_app_folder_path, "static", "predicted_images", image_file.filename))
        return render_template("index.html", uploaded_path="uploaded_images/"+image_file.filename,
                                    predicted_path="predicted_images/"+image_file.filename )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
