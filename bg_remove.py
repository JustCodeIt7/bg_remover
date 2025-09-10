# --- Step 1: Import Necessary Libraries ---
# In this step, we import all the libraries we'll need for the app.
# Streamlit for the UI, rembg for background removal, PIL for image processing,
# and others for handling bytes, base64, OS operations, error tracing, and time.
import streamlit as st
from rembg import remove
from PIL import Image
import numpy as np  # Note: This import is not used in the code; consider removing if unnecessary
from io import BytesIO
import base64
import os
import traceback
import time

# --- Step 2: Set Up Page Configuration ---
# Here, we configure the Streamlit page layout and title.
# This makes the app wide and sets a custom page title.
st.set_page_config(layout="wide", page_title="Image Background Remover")

# --- Step 3: Define Constants ---
# Define constants for file size limits and image processing.
# These can be easily adjusted if needed.
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB maximum upload size
MAX_IMAGE_SIZE = 2000  # Maximum dimension in pixels for processing

# --- Step 4: Helper Functions ---


# Function to convert an image to bytes for downloading.
# This prepares the image in PNG format for the download button.
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


# Function to resize an image while maintaining aspect ratio.
# This prevents memory issues with very large images.
def resize_image(image, max_size):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.LANCZOS)


# Cached function to process the image.
# Caching avoids reprocessing the same image multiple times.
@st.cache_data
def process_image(image_bytes):
    """Process image with caching to avoid redundant processing"""
    try:
        image = Image.open(BytesIO(image_bytes))
        # Resize large images to prevent memory issues
        resized = resize_image(image, MAX_IMAGE_SIZE)
        # Remove background using rembg
        fixed = remove(resized)
        return image, fixed
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None


# Main function to handle image fixing and UI updates.
# This orchestrates the processing, progress updates, and display.
def fix_image(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading image...")
        progress_bar.progress(10)

        # Read image bytes from upload or file path
        if isinstance(upload, str):
            # Default image path
            if not os.path.exists(upload):
                st.error(f"Default image not found at path: {upload}")
                return
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            # Uploaded file
            image_bytes = upload.getvalue()

        status_text.text("Processing image...")
        progress_bar.progress(30)

        # Process the image using the cached function
        image, fixed = process_image(image_bytes)
        if image is None or fixed is None:
            return

        progress_bar.progress(80)
        status_text.text("Displaying results...")

        # Display original and processed images in columns
        col1.write("Original Image :camera:")
        col1.image(image)

        col2.write("Fixed Image :wrench:")
        col2.image(fixed)

        # Add download button in sidebar
        st.sidebar.markdown("\n")
        st.sidebar.download_button(
            "Download fixed image", convert_image(fixed), "fixed.png", "image/png"
        )

        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process image")
        # Log the full error for debugging (visible in console)
        print(f"Error in fix_image: {traceback.format_exc()}")


# --- Step 5: Set Up the User Interface ---
# Display the main title and description.
st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)

# Sidebar setup for upload and download.
st.sidebar.write("## Upload and download :gear:")

# Create columns for displaying images side by side.
col1, col2 = st.columns(2)

# File uploader in the sidebar.
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Expander for image guidelines in the sidebar.
with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    - Processing time depends on image size
    """)

# --- Step 6: Process the Image ---
# Handle uploaded image or load default if none uploaded.
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(
            f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE / 1024 / 1024:.1f}MB."
        )
    else:
        fix_image(upload=my_upload)
else:
    # Try loading default images if no upload.
    default_images = ["./zebra.jpg", "./wallaby.png"]
    for img_path in default_images:
        if os.path.exists(img_path):
            fix_image(img_path)
            break
    else:
        st.info("Please upload an image to get started!")
