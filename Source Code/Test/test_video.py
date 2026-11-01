import cv2
import os

video_path = "test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video is not open")
    raise SystemExit

print("Video opened")

## ----------------------------------------------------
frame_num = 0
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

while True:
    success, frame = cap.read()
    frame_num += 1
    
    if not success:
        print(f"{frame_num:04d} frame not saved")
        break

    output_folder = "Frames"
    os.makedirs(output_folder, exist_ok = True)

    filename = os.path.join(output_folder, f"frame_{frame_num:04d}.png")

    saved = cv2.imwrite( filename, frame)

    print(f"{frame_num:04d} frame saved")

print("Total Frames Saved:", frame_num)

# ----------------------------------------------------
##width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#fps = cap.get(cv2.CAP_PROP_FPS)
#frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

#print("Wigth:", width)
#print("Height:", height)
#print("FPS:", fps)
#print("Frame Count:", frame_count)

# ----------------------------------------------------
#frame_count = 0

#while True:
#    success, frame = cap.read()
#    frame_count += 1

#    if not success:
#        break

#print("Total Frames Read:", frame_count)

# ----------------------------------------------------
#if success:
#    print("Frame read successfully")
#    print("Frame shape:", frame.shape)
#else:
#    print("Error: Could not read the first frame")

cap.release()