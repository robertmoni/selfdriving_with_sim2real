import argparse
import glob
import os.path

import numpy as np

import cv2
import wget
from tqdm import tqdm

urls = list()


def download_videos(download_path):
    print("Downloading videos...")
    for i in tqdm(range(len(urls))):
        file_path = os.path.join(download_path, f'{i:03d}.mp4')
        wget.download(urls[i], file_path)
    print("Downloading finished.")


def saveAsImages(save_path):
    files = sorted(glob.glob(os.path.join(save_path, '*.mp4')))
    # files = sorted(os.path.join(save_path, '*.mp4'))
    # imgDir = os.path.join(args.save_path, "train")
    print("Saving as images...")
    saved_ims = 0
    for file in tqdm(files):
        cap = cv2.VideoCapture(file)
        
        skip_no_frame = True
        while cap.isOpened():
            ret, frame = cap.read()
            # skip every second frame
            if skip_no_frame:
                if ret and frame is not None:
                    frame = cv2.resize(frame, dsize=(64,64), interpolation = cv2.INTER_AREA)
                    filename = os.path.join(save_path, f"{saved_ims:06d}.png")
                    cv2.imwrite(filename, frame)
                    #np.save(filename, np.asarray(frame))
                    saved_ims += 1
                    skip_no_frame = False
                else:
                    break
            else:
                skip_no_frame = True


        cap.release()
        os.remove(file)
    print("Saving as images finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="/selfdriving_with_sim2real/data/train/real")
    parser.add_argument('--as_videos', action='store_true')
    args = parser.parse_args()

    scriptDir = os.path.split(__file__)[0]
    urlFile = os.path.join(scriptDir, "realVideoURLs.txt")
    with open(urlFile, 'r') as file:
        for url in file:
            if len(url) > 0:
                urls.append(url.rstrip())

    # os.makedir(args.save_path, exist_ok=True)
    
    download_videos(args.save_path)
    if not args.as_videos:
        saveAsImages(args.save_path)


#         import argparse
# import glob
# import os.path

# import numpy as np

# import cv2
# import wget
# from tqdm import tqdm

# urls = list()


# def download_videos(download_path):
#     print("Downloading videos...")
#     for i in tqdm(range(len(urls))):
#         file_path = os.path.join(download_path, f'{i:03d}.mp4')
#         wget.download(urls[i], file_path)
#     print("Downloading finished.")


# def saveAsImages(save_path):
#     files = sorted(glob.glob(os.path.join(save_path, '*.mp4')))
#     # imgDir = os.path.join(args.save_path, "train")
#     print("Saving as images...")
#     saved_ims = 0
#     for file in tqdm(files):
#         cap = cv2.VideoCapture(file)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if ret and frame is not None:
                
#                 filename = os.path.join(args.save_path, f"{saved_ims:06d}.npy")
#                 # cv2.imwrite(filename, frame)
#                 np.save(filename, np.asarray(frame))
#                 saved_ims += 1
#             else:
#                 break

#         cap.release()
#         os.remove(file)
#     print("Saving as images finished.")
