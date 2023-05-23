from glob import glob
import os

from dotenv import load_dotenv
import numpy as np
import scipy.io as sio

load_dotenv()

# PATH = '/Users/m/workspace/phd/datasets/300W_LP/AFW'
DATASET_PATH = os.getenv("DATASET_PATH")

print(f'Loaded PATH: {DATASET_PATH}')

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def convert_radians_to_degrees(math_path):
    pose = get_ypr_from_mat(math_path)
    pitch = (pose[0] * 0.5) / np.pi + 1
    yaw = (pose[1] * 0.5) / np.pi + 1
    roll = (pose[2] * 0.5) / np.pi + 1

    pitch = pitch / 2
    yaw = yaw / 2
    roll = roll / 2

    # pitch = pose[0]
    # yaw = pose[1]
    # roll = pose[2]

    # pitch = (pose[0] * 180) / np.pi + 90
    # yaw = (pose[1] * 180) / np.pi + 90
    # roll = (pose[2] * 180) / np.pi + 90

    yaw, pitch, roll = pose[0], pose[1], pose[2]

    return yaw, pitch, roll

def getInputsList(directory):
    # Only files directly placed in the directory path are loaded
    # @TODO: Recursively read directory
    imageExtensions = ['jpg', 'png', 'jpeg']
    if os.path.exists(directory):
        matFilePaths = glob(os.path.join(directory.rstrip('/'), '*.mat'))
        imageFilePaths = []
        for imageExtension in imageExtensions:
            upperCaseExtensions = list(
                set(
                    glob(
                        os.path.join(
                            directory.rstrip('/'),
                            f'*.{imageExtension.upper()}'
                        )
                    )
                )
            )
            lowerCaseExtensions = list(
                set(
                    glob(
                        os.path.join(
                            directory.rstrip('/'),
                            f'*.{imageExtension.lower()}'
                        )
                    )
                )
            )                      
            imageFilePaths += upperCaseExtensions + lowerCaseExtensions
    
    return matFilePaths, imageFilePaths

def organizeInputSamples(matFiles, imageFiles):
    imageFileNames = {os.path.splitext(os.path.basename(imageFile))[0]: imageFile for imageFile in imageFiles}
    inputSamples = {}

    for matFile in matFiles:
        basename = os.path.basename(matFile)
        name, extension = os.path.splitext(basename)
        extension = extension.lstrip('.')
        if name in imageFileNames:
            inputSamples[name] = [matFile, imageFileNames.get(name)]

    return inputSamples

def loadAndPlotImageYpr(path):
    matFiles, imageFiles = getInputsList(path)
    inputSamples = organizeInputSamples(matFiles, imageFiles)

    inputData = []
    for key, files in inputSamples.items():
        yaw, pitch, roll = convert_radians_to_degrees(files[0])
        image = files[1]
        inputData.append((image, (yaw, pitch, roll, ), ))
    
    return inputData