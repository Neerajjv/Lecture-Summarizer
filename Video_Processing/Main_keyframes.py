import os
import cv2
import shutil
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Softwares\tessaract\tesseract.exe"


def process_video_and_cluster(video_path, frame_rate, output_slides_dir):

    def clear_directory(directory):
        """Clear the contents of a directory."""
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    def get_frames(video_path, frame_rate):
        """Extract frames at every frame_rate seconds."""
        vs = cv2.VideoCapture(video_path)
        if not vs.isOpened():
            raise Exception(f"Unable to open video {video_path}")

        total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_time = 0
        frame_count = 0
        print(f"Total frames in video: {total_frames}")

        while True:
            vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            ret, frame = vs.read()
            if not ret:
                break
            frame_count += 1
            frame_time += frame_rate
            yield frame_count, frame_time, frame

        vs.release()

    def save_frames(video_path, output_folder, frame_rate):
        """Extract frames and save to output folder at every frame_rate seconds."""
        clear_directory(output_folder)

        for frame_count, _, frame in get_frames(video_path, frame_rate):
            frame_name = f"{frame_count:03}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

        print(f"Frames saved to {output_folder}")

    def extract_features_from_images(image_folder):
        """Extract features using a pre-trained CNN (ResNet50)."""
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        features = []
        image_paths = []

        for img_name in sorted(os.listdir(image_folder)):
            img_path = os.path.join(image_folder, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image = preprocess_input(image)
            features.append(image)
            image_paths.append(img_path)

        features = np.array(features)
        features = model.predict(features)
        return features, image_paths

    def cluster_images(features, image_paths, representative_output_folder):
        """Cluster images and save all representative images in one folder."""
        clear_directory(representative_output_folder)
        features_scaled = StandardScaler().fit_transform(features)

        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2)
        labels = clusterer.fit_predict(features_scaled)

        # Save one representative frame from each cluster
        saved_representatives = set()  # To avoid duplicates
        for label in set(labels):
            if label == -1:
                continue  # Skip noise
            cluster_images = [image_paths[i] for i in range(len(labels)) if labels[i] == label]
            representative_image = cluster_images[0]
            if representative_image not in saved_representatives:
                shutil.copy(representative_image, os.path.join(representative_output_folder, os.path.basename(representative_image)))
                saved_representatives.add(representative_image)



    # Define output folders
    output_frames_dir = os.path.join(output_slides_dir, "extracted_frames")
    cluster_representative_dir = os.path.join(output_slides_dir, "cluster_representatives")

    # Step 1: Extract frames from the video
    save_frames(video_path, output_frames_dir, frame_rate)

    # Step 2: Extract features from the saved frames
    features, image_paths = extract_features_from_images(output_frames_dir)

    # Step 3: Cluster the frames and save representative frames
    cluster_images(features, image_paths, cluster_representative_dir)

    # Step 4: Detect and save diagrams from representative frames

    print(f"Processing complete. Representatives saved in {cluster_representative_dir}.")
    return {
        "representative_frames": cluster_representative_dir,

    }


# Example usage
if __name__ == "__main__":
    video_path = "test.mp4"  # Change this to your video path
    frame_rate = 5  # 1 frame every 5 seconds
    output_slides_dir = "./output"

    output_dirs = process_video_and_cluster(video_path, frame_rate, output_slides_dir)
    print(f"Representative frames directory: {output_dirs['representative_frames']}")
