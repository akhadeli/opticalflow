import cv2

cam = cv2.VideoCapture(0)
counter = 0

print('Capturing')
while counter < 30:
    ret, frame = cam.read()
    cv2.imwrite(f'images/{counter+1}.png', frame)
    counter += 1
