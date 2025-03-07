import cv2 
import numpy as np 
import time 
  
# in order to check the cv2 version 
print(cv2.__version__)    
  
# Capture the video from a file or webcam
capture_video = cv2.VideoCapture(0)  # or 0 for webcam
     
# Give the camera some time to warm up
time.sleep(1)  
count = 0 
background = 0 
  
# Capturing the background (assuming a video with some background frames)
for i in range(60): 
    return_val, background = capture_video.read() 
    if not return_val:
        continue 
  
background = np.flip(background, axis=1)  # Flip background horizontally
  
# Process the video frames
while capture_video.isOpened(): 
    return_val, img = capture_video.read() 
    if not return_val:
        break 
    count += 1
    img = np.flip(img, axis=1)  # Flip the frame horizontally
  
    # Convert the image to HSV color space for better detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
  
    #----------------------------------BLOCK FOR BLACK DETECTION---------------------------# 
    # Set the HSV range for black color
    lower_black = np.array([0, 0, 0])       # Lower bound for black
    upper_black = np.array([180, 255, 50])  # Upper bound for black (adjust if necessary)
    
    # Create a mask to detect black areas in the frame
    mask1 = cv2.inRange(hsv, lower_black, upper_black)
    #-------------------------------------------------------------------------------# 
  
    # Refine the mask for black color detection
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2) 
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1) 
    mask2 = cv2.bitwise_not(mask1)  # Invert the mask
  
    # Generate the output by combining the background and current frame
    res1 = cv2.bitwise_and(background, background, mask=mask1)  # Background where black is
    res2 = cv2.bitwise_and(img, img, mask=mask2)  # Original frame where black is not detected
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0) 
  
    # Display the final result
    cv2.imshow("Harry Potter", final_output) 
    
    # Exit when 'Esc' key is pressed
    if cv2.waitKey(10) == 27:  # 'Esc' key
        break
  
# Release resources and close windows
capture_video.release()
cv2.destroyAllWindows()
