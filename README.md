# Harry-Potter-Effect /  Black Color Cloak Effect Using OpenCV

## Overview
This project implements a **Harry Potter invisibility cloak effect** using OpenCV in Python. It detects the black color in the frame and replaces it with a previously captured background, creating an illusion of invisibility.

## Features
- Captures live video from a webcam.
- Detects black color in the frame.
- Replaces black-colored areas with the background.
- Displays the processed video with an invisibility effect.
- Exits when the **Esc** key is pressed.

## Prerequisites
Ensure you have Python installed along with the necessary dependencies:

```bash
pip install opencv-python numpy
```

## How It Works
1. The script initializes the webcam and captures a static background frame.
2. It continuously reads frames from the webcam.
3. The frames are converted to **HSV color space** to detect the black color.
4. A mask is created to filter out black areas.
5. The black areas are replaced with the stored background.
6. The processed frame is displayed in real-time.

## Code Breakdown
### **1. Import Required Libraries**
```python
import cv2
import numpy as np
import time
```
- **cv2** for image processing
- **numpy** for numerical operations
- **time** to allow camera warm-up

### **2. Initialize Camera and Capture Background**
```python
capture_video = cv2.VideoCapture(0)  # Use 0 for webcam

# Allow the camera to warm up
time.sleep(1)  
background = 0

# Capture the background frame
for i in range(60):
    return_val, background = capture_video.read()
    if not return_val:
        continue

background = np.flip(background, axis=1)  # Flip horizontally
```
- Initializes webcam capture.
- Gives the camera time to adjust.
- Captures a reference **background** frame for later use.
- **Flips** the background frame horizontally to match real-world orientation.

### **3. Process Video Frames**
```python
while capture_video.isOpened():
    return_val, img = capture_video.read()
    if not return_val:
        break
    img = np.flip(img, axis=1)  # Flip the frame horizontally
```
- Reads video frames.
- **Flips** each frame to maintain correct alignment.

### **4. Convert Frame to HSV Color Space**
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
```
- Converts **BGR** frame to **HSV** for better color detection.

### **5. Create Mask for Black Color Detection**
```python
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
mask1 = cv2.inRange(hsv, lower_black, upper_black)
```
- Defines **HSV range** for detecting black color.
- Creates a **binary mask** where black pixels are detected.

### **6. Refine the Mask**
```python
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
mask2 = cv2.bitwise_not(mask1)  # Invert the mask
```
- Removes noise using **morphological operations**.
- **Dilates** the mask to cover more black areas.
- Inverts the mask to separate non-black regions.

### **7. Replace Black Areas with Background**
```python
res1 = cv2.bitwise_and(background, background, mask=mask1)  # Background where black is
res2 = cv2.bitwise_and(img, img, mask=mask2)  # Original frame where black is not detected
final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
```
- **Combines** background and original frame.
- Keeps original content where black is **not detected**.

### **8. Display the Final Output**
```python
cv2.imshow("Harry Potter", final_output)
```
- Displays the processed video frame.

### **9. Exit on 'Esc' Key**
```python
if cv2.waitKey(10) == 27:  # 'Esc' key
    break
```
- Allows exiting the program when **Esc** is pressed.

### **10. Release Resources**
```python
capture_video.release()
cv2.destroyAllWindows()
```
- Releases the webcam.
- Closes all OpenCV windows.

Wear **black clothes**, and the script will replace black areas with the captured background, creating an **invisibility cloak effect**.

## Customization
- Adjust `upper_black` values in HSV to fine-tune black color detection.
- Experiment with different backgrounds.
- Modify `cv2.dilate()` to improve mask detection.



