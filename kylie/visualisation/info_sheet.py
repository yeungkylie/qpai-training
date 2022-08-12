import cv2
import numpy as np
import os

SET_NAME = "Baseline"
processes = [None, "thresholded", "smoothed", "noised", "thresholded_smoothed"]
process_name = ["No Processing",
                "Thresholded",
                "Smoothed",
                "Noised",
                "Thresholded and Smoothed"]
font = cv2.FONT_HERSHEY_DUPLEX

def combine_processing_plots(SET_NAME):
    img=[[] for i in range(len(processes))]
    for process in enumerate(processes):
        print(process[0])
        FOLDER = f"I:/research\seblab\data\group_folders\Kylie\images\spectra/"
        if process[1] is None:
            FILE =f"{SET_NAME}.png"
        else:
            FILE = f"{SET_NAME}_{process[1]}.png"
        temp_img = cv2.imread(os.path.join(FOLDER,FILE))
        scale_percent = 40  # percent of original size
        width = int(temp_img.shape[1] * scale_percent / 100)
        height = int(temp_img.shape[0] * scale_percent / 100)
        cv2.putText(temp_img, process_name[process[0]], (20, 30), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        img[process[0]] = cv2.resize(temp_img,(width,height))
    h1, w1 = img[0].shape[:2]

    img_3 = np.zeros((h1*5, w1,3), dtype=np.uint8)
    img_3[:,:] = (255,255,255)

    for i in range(len(processes)):
        img_3[h1*i:h1*(i+1), :w1,:3] = img[i]

    # cv2.imshow('Img_1',img_1)
    # cv2.imshow('Img_2',img_2)
    cv2.imshow('Img_3',img_3)


def spectra_and_pca(SET_NAME):
    spectra_file = f"I:/research\seblab\data\group_folders\Kylie\images\spectra/{SET_NAME}.png"
    pca_file = f"I:/research\seblab\data\group_folders\Kylie\images\pca/{SET_NAME}_pca.png"
    temp_img = cv2.imread(spectra_file)
    scale_percent = 60  # percent of original size
    height = int(temp_img.shape[0] * scale_percent / 100)
    width = int(temp_img.shape[1] * scale_percent / 100)
    img_1 = cv2.resize(temp_img, (width, height))
    img_2 = cv2.imread(pca_file)

    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]

    img_3 = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
    img_3[:, :] = (255, 255, 255)

    img_3[:h1, :w1, :3] = img_1
    img_3[h1:h1 + h2, :w2, :3] = img_2

    # cv2.imshow('Img_1', img_1)
    # cv2.imshow('Img_2', img_2)
    cv2.imshow('Img_3', img_3)
    cv2.waitKey(0)


if __name__ == "__main__":
    spectra_and_pca("0.6mm Res")