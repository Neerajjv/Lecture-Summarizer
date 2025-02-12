import os
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Softwares\tessaract\tesseract.exe"
def extract_text_from_images(input_dir, output_file):
    """
    Processes all images in the specified directory, extracts text using OCR,
    and saves the result to a single text file.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # Clear the output file at the start
    open(output_file, "w").close()

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        print(f"Processing {img_name}...")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OTSU Threshold
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Create a rectangular kernel for dilation
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Dilate the thresholded image
        dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Make a copy of the original image
        image_copy = image.copy()

        # Open the file for appending extracted text
        with open(output_file, "a", encoding="utf-8") as file:
            for contour in contours:
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the region of interest
                cropped = image_copy[y:y + h, x:x + w]

                # Perform OCR on the cropped image
                text = pytesseract.image_to_string(cropped)

                # Write the text to the file
                file.write(text)
                file.write("\n")

    print(f"All text extracted and saved to {output_file}.")