import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detect_changes(before_img, after_img):
    """Detect changes between two images and return both the difference and marked image."""
    # Convert to numpy arrays if they're PIL Images
    before = np.array(before_img)
    after = np.array(after_img)
    
    # Convert to grayscale for processing
    gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    
    # Resize to same dimensions
    height = min(gray_before.shape[0], gray_after.shape[0])
    width = min(gray_before.shape[1], gray_after.shape[1])
    
    gray_before = cv2.resize(gray_before, (width, height))
    gray_after = cv2.resize(gray_after, (width, height))
    before = cv2.resize(before, (width, height))
    after = cv2.resize(after, (width, height))
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray_before, gray_after)
    
    # Apply threshold to create a binary mask
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the 'after' image to draw rectangles on
    marked = after.copy()
    
    # Draw rectangles around changed areas
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red rectangle
    
    # Convert difference to 3-channel for display
    diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    
    return marked, diff_colored, len(contours)

def main():
    st.title("🖼️ Image Change Detector")
    st.write("Upload two images to compare them and detect changes.")
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Image")
        before_file = st.file_uploader("Choose first image", type=["jpg", "png", "jpeg"], key="before")
        if before_file:
            st.image(before_file, use_column_width=True)
    
    with col2:
        st.subheader("After Image")
        after_file = st.file_uploader("Choose second image", type=["jpg", "png", "jpeg"], key="after")
        if after_file:
            st.image(after_file, use_column_width=True)
    
    if before_file and after_file:
        if st.button("🔍 Detect Changes", type="primary"):
            with st.spinner("Analyzing images..."):
                try:
                    # Open and process images
                    img1 = Image.open(before_file).convert('RGB')
                    img2 = Image.open(after_file).convert('RGB')
                    
                    # Detect changes
                    marked_img, diff_img, num_changes = detect_changes(img1, img2)
                    
                    # Display results in two columns
                    st.success("Analysis complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Marked Changes")
                        st.image(marked_img, use_column_width=True, clamp=True)
                        st.caption("Red rectangles highlight detected changes")
                    
                    with col2:
                        st.subheader("Difference Map")
                        st.image(diff_img, use_column_width=True, clamp=True)
                        st.caption("White areas show where changes were detected")
                    
                    # Show summary
                    st.metric("Number of Changes Detected", num_changes)
                    
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")

if __name__ == "__main__":
    main()
