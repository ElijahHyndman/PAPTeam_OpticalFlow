# Author: Elijah Hyndman 2020

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
#from PIL import Image
import time
import argparse
import pyflow
import os
import cv2
try: # import our own local made modules
    from InputCreation.TestImagePairGenerator import TestImagePairGenerator
    from InputCreation.TestImagePair import TestImagePair
    from InputCreation.TestImage import TestImage
    from InputCreation.ImageCollection import ImageCollection
except Exception as e:
     print('*'*3+'Input Creation Modules were not imported: {}'.format(e))

# Parsing Arguments
'''
parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()
'''


'''
   ____        _   _           _   ______ _
  / __ \      | | (_)         | | |  ____| |
 | |  | |_ __ | |_ _  ___ __ _| | | |__  | | _____      __
 | |  | | '_ \| __| |/ __/ _` | | |  __| | |/ _ \ \ /\ / /
 | |__| | |_) | |_| | (_| (_| | | | |    | | (_) \ V  V /
  \____/| .__/ \__|_|\___\__,_|_| |_|    |_|\___/ \_/\_/
        | |
        |_|


    This file is a python wrapper for the cpp implementation of Optical Flow.
        The cpp functions are called using the pyflow.pyx cython file
'''

def CalculateOpticalFlow(imagePair, pyramidLevels, numCores, optionalOutputSuffix=''):
    ''' CalculateOpticalFlow calculates the optical flow for two TestImage objects, \
            image_A and image_B

            :param imagePair: TestImagePair that will be used to calculate
                                Optical Flow
            :param pyramidLevels: an integer height for the LK compression pyramid
            :param numcores: an integer for the number of hardware cores/threads to use for calculation
    '''
    # Begin Timer
    start=time.perf_counter()

    # Debug
    image_A=imagePair.BEFORE
    image_B=imagePair.AFTER

    # === Create Image Arrays
    im1=np.array(Image.open(image_A.IMAGE_PATH))
    im2=np.array(Image.open(image_B.IMAGE_PATH))
    imDimensions=im1.shape
    # Normalize 0-255 to 0-1
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # === Calculation
    timingDictionary, u, v, im2W = pyflow.coarse2fine_flow(im1, im2, pyramidLevels, numCores)
    # === Output Vectors to array
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    debugTimingDiagnostics(timingDictionary)
    # === DEBUG: print the timing contents of this calculation to terminal
    #print(timingDictionary.items())

    # === All of the output Storing
    generateOutput(imagePair, pyramidLevels, numCores, flow, imDimensions, timingDictionary, optionalOutputSuffix)

def debugTimingDiagnostics(timingDictionary):
    # Copy values to a dictionary that we can edit freely
    timing=timingDictionary.copy()
    # Items that will be displayed separately from the rest of the timing dictionary
    separatedKeys=('Total C++ Execution','Total Flow Calculation')
    # Display the separatedKeys
    for item in separatedKeys:
        print('--'+item+': ',timing.pop(item))
    # Display the rest
    print(timing)


def generateOutput(imagePair, nLevels, nCores, flow, imDimensions, timingDictionary, optionalOutputSuffix):
    ''' Generate all output relevant to Optical Flow

        :param imagePair: TestImagePair object that was used to create \
                            Optical Flow output
        :param flow: np array of (u,v) coordinates for vectors
        :param im1: np array created from one of the JPG RGB input images
    '''

    # === Folder Names for Path
    # All output will be stored in ./output
    OUTPUT_DIRECTORY=os.path.join(os.getcwd(),'output')
    # This output is stored in ./output/ImageCollectionName
    COLLECTION_OUTPUT_FOLDER_NAME=imagePair.BEFORE.IMAGE_PARENT+optionalOutputSuffix
    OutputFolder=os.path.join(OUTPUT_DIRECTORY,COLLECTION_OUTPUT_FOLDER_NAME)
    # images and timing are stored in separate respective folders
    Image_OutputFolder=os.path.join(OutputFolder,'images_P{}'.format(nLevels))
    Timing_OutputFolder=os.path.join(OutputFolder,'timing')

    # Create said Directories if they don't exist
    os.makedirs(OutputFolder,exist_ok=True)
    os.makedirs(Image_OutputFolder,exist_ok=True)
    os.makedirs(Timing_OutputFolder,exist_ok=True)

    # === Files
    # Image File Name
    # Output image names are derived from first input image
    Frame_Base=('frame{}'.format(imagePair.BEFORE.IMAGE_INDEX_STRING))
    Image_OutputExtension='.jpg'
    Image_OutputFile=Frame_Base+Image_OutputExtension
    OUTPUT_IMAGE_PATH=os.path.join(Image_OutputFolder,Image_OutputFile)

    # Timing File Name
    Timing_OutputFile=COLLECTION_OUTPUT_FOLDER_NAME+'_P{}_C{}.txt'.format(nLevels,nCores)
    OUTPUT_TIMING_PATH=os.path.join(Timing_OutputFolder,Timing_OutputFile)

    # === Create the final output files
    generateOutputFlowImageFile(OUTPUT_IMAGE_PATH,flow,imDimensions)
    generateOutputTimingFile(OUTPUT_TIMING_PATH,timingDictionary)



def generateOutputFlowImageFile(outputImagePath,flow, imDimensions):
    ''' Creates output image file from the flow vector

            :param outputImagePath: the absolute path to the desired output \
                                        image (image file name inclusive.)
            :param flow: numpy array of (u,v) vectors
            :param imDimensions: dimensions of the output image
    '''
    # Generate Black Image
    hsv = np.zeros(imDimensions, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    # Convert (u,v) vectors to magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert magnitude and angle into intensity and hue
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Write image data to image file
    cv2.imwrite(outputImagePath, rgb)

    #print('stored output image at:',outputImagePath)



def generateOutputTimingFile(outputFilePath, timingDictionary):
    # We will write the header first, if the file does not exist yet
    writeHeader= not os.path.exists(outputFilePath)
    f = open(outputFilePath,'a')
    delimiter=','

    if( writeHeader ):
        HeaderString=delimiter.join( timingDictionary.keys() )
        f.write(outputFilePath+'\n')
        f.write(HeaderString+'\n')

    # Join all of the values in the timingDictionary, delimit with delimiter string
    timingString=delimiter.join( timingDictionary.values() )
    f.write(timingString+'\n')
    f.close()
