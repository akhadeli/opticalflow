import cv2
import numpy as np
import os

IMAGE_FOLDER = 0
VIDEO_MODE = 1
LIVE_MODE = 2
DIFFERENCES_MODE = 3

def difference(img1, img2):
    A = np.subtract(img2, img1, dtype=np.float32)
    # A[np.abs(A) <= 10] = 0
    return A

def threshold(img, threshold=0):
    img[np.abs(img) <= threshold] = 0
    return img

def lowpassFilter(img, kernelSize, sigma=1.0, mu=0.0):
    x, y = np.meshgrid(np.linspace(-1, 1, kernelSize), np.linspace(-1, 1, kernelSize))
    d = np.sqrt(x*x + y*y)
    kernel = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

    smoothed = cv2.filter2D(img, -1, kernel)
    return smoothed

class OpticalFlow:
    def __init__(self, mode=IMAGE_FOLDER, save_output=False, output_filename='output.avi', **kwargs):
        self.save = save_output
        self.output_filename = output_filename
        
        if mode == IMAGE_FOLDER or mode == DIFFERENCES_MODE:
            self.path = kwargs['path']  
            assert os.path.isdir(self.path)
            self.file_list = [self.path+file for file in sorted(os.listdir(self.path), key=lambda f: int(os.path.splitext(f)[0]))]
            assert len(self.file_list) % 2 == 0

        elif mode == LIVE_MODE:
            self.source = kwargs['source']
            self.threshold_value = 40
            self.blur = False
            self.kernelSize = 10
            self.tile_size = 16
            self.arrow_scale = 5
            self.writer = None
            self.debug = False
            self.diff_debug = False
            self.liveProcess()


    def detect(self, tile_thresholded, tile):
        if np.any(tile_thresholded):
            Iy, Ix = np.gradient(np.float64(tile))
            return (Iy, Ix)
        else:
            return 0
    
    def drawVector(self, frame, j, k, tile_size, vectors, scale):
        # vertical, horizontal
        centre = (int(k + tile_size/2), int(j + tile_size/2))

        arrow = (int(centre[0] + scale*vectors[0]), int(centre[1] + scale*vectors[1]))
        # print(centre, arrow)
        cv2.arrowedLine(frame, centre, arrow, [255, 0, 0], 1)

    def drawGrid(self, frame, j, k, tile_size):
        centre = (int(k + tile_size/2), int(j + tile_size/2))
        cv2.circle(frame, centre, radius=0, color=(0, 0, 0), thickness=1)

    def process(self, tile_size, arrow_scale, threshold_value, blur=False, kernelSize=10, debug=False):
        i = 1
        img1_o = cv2.imread(self.file_list[0])

        if self.save:
            writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (img1_o.shape[1], img1_o.shape[0]))

        while i < len(self.file_list):
            # img1_o = cv2.imread(self.file_list[i])
            img2_o = cv2.imread(self.file_list[i])

            img1 = cv2.cvtColor(img1_o, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2_o, cv2.COLOR_BGR2GRAY)

            output_frame = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            differences = difference(img1, img2)
            thresholded_frame = threshold(differences, threshold_value)

            if debug:
                cv2.imshow('thresh', thresholded_frame)
            
            if blur:
                thresholded_frame = lowpassFilter(thresholded_frame, kernelSize)
    
            if debug and blur:
                cv2.imshow('lowpass', thresholded_frame)

            height, width = thresholded_frame.shape
            
            # height equivalent to number of rows
            # width equivalent to number of columns
            for j in range(0, height, tile_size):
                for k in range(0, width, tile_size):
                    tile_d = thresholded_frame[j:j+tile_size, k:k+tile_size]
                    tile_i = img1[j:j+tile_size, k:k+tile_size]
                    self.drawGrid(output_frame, j, k, tile_size)
                    detection = self.detect(tile_d, tile_i)
                    if detection != 0:
                        tile_d = tile_d.flatten().T
                        dI = np.array([detection[0].flatten(), detection[1].flatten()]).T
                        x = np.linalg.lstsq(dI, tile_d, rcond=None)
                        self.drawVector(output_frame, j, k, tile_size, x[0], arrow_scale)


            cv2.imshow('output', output_frame)
            if self.save:
                writer.write(output_frame)

            img1_o = cv2.imread(self.file_list[i])
            i += 1

            key = cv2.waitKey(30)
            if key == 27:
                break
        
        cv2.destroyAllWindows()
    
    def liveProcess(self):
        cap = cv2.VideoCapture(0, self.source)

        ret, frame0 = cap.read()

        if not ret:
            raise Exception("Failed to read from capture device")

        # Saving first frame
        frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)


        if self.save:
            self.writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame0.shape[1], frame0.shape[0]))
        
        if self.diff_debug:
            self.diff_writer = cv2.VideoWriter('diff_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame0.shape[1], frame0.shape[0]), False)

        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to read from capture device")
            
            output_frame = cv2.cvtColor(frame0_gray, cv2.COLOR_GRAY2BGR)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            differences = difference(frame0_gray, frame_gray)
            thresholded_frame = threshold(differences, threshold=self.threshold_value)

            if self.debug:
                cv2.imshow('thresh', thresholded_frame)

            if self.blur:
                thresholded_frame = lowpassFilter(thresholded_frame, self.kernelSize)

            height, width = thresholded_frame.shape

            # height equivalent to number of rows
            # width equivalent to number of columns
            for j in range(0, height, self.tile_size):
                for k in range(0, width, self.tile_size):
                    tile_d = thresholded_frame[j:j+self.tile_size, k:k+self.tile_size]
                    tile_i = frame0_gray[j:j+self.tile_size, k:k+self.tile_size]
                    self.drawGrid(output_frame, j, k, self.tile_size)
                    detection = self.detect(tile_d, tile_i)
                    if detection != 0:
                        tile_d = tile_d.flatten().T
                        dI = np.array([detection[0].flatten(), detection[1].flatten()]).T
                        x = np.linalg.lstsq(dI, tile_d, rcond=None)
                        self.drawVector(output_frame, j, k, self.tile_size, x[0], self.arrow_scale)

            cv2.imshow('output', output_frame)
            if self.save:
                self.writer.write(output_frame)

            if self.diff_debug:
                self.diff_writer.write(thresholded_frame)

            frame0_gray = frame_gray

            key = cv2.waitKey(20)
            if key == 27:
                break
        
if __name__ == "__main__":
    # Image folder demo
    o = OpticalFlow(IMAGE_FOLDER, save_output=True, path='./armD32im1/')
    o.process(tile_size=16, arrow_scale=2, threshold_value=40, blur=True, kernelSize=5, debug=False)

    # Video demo

    # Live demo
    # l = OpticalFlow(LIVE_MODE, source=0, save_output=True)