# Author: Deepak Pathak (c) 2016
# Edited: Elijah Hyndman 2020

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
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
parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()



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

def CalculateOpticalFlow(imagePair, pyramidLevels, numCores):
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

    print(timingDictionary,'\n')
    # === All of the output Storing
    generateOutput(imagePair, pyramidLevels, numCores, flow, imDimensions, timingDictionary)


def generateOutput(imagePair, nLevels, nCores, flow, imDimensions, timingDictionary):
    ''' Generate all output relevant to Optical Flow

        :param imagePair: TestImagePair object that was used to create \
                            Optical Flow output
        :param flow: np array of (u,v) coordinates for vectors
        :param im1: np array created from one of the JPG RGB input images
    '''

    # === Folder Names for Path
    # All output will be stored in ./output
    COLLECTION_OUTPUT_DIRECTORY=os.path.join(os.getcwd(),'output')
    # This output is stored in ./output/ImageCollectionName
    OUTPUT_FOLDER_NAME=imagePair.BEFORE.IMAGE_PARENT
    OutputFolder=os.path.join(COLLECTION_OUTPUT_DIRECTORY,OUTPUT_FOLDER_NAME)
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
    Timing_OutputFile=OUTPUT_FOLDER_NAME+'_P{}_C{}.txt'.format(nLevels,nCores)
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
    writeHeader= not os.path.exists(outputFilePath)
    f = open(outputFilePath,'a')
    delimiter=','

    # If File does not exist, write in the Header String
    if( writeHeader ):
        HeaderString=delimiter.join( timingDictionary.keys() )
        f.write(outputFilePath+'\n')
        f.write(HeaderString+'\n')

    timingString=delimiter.join( timingDictionary.values() )
    f.write(timingString+'\n')
    f.close()















# Test lines of code to run the functions

'''
# Create Test Image data
pairGenerator=TestImagePairGenerator()
pairGenerator.COLLECTION_KEYS.sort()
# Output All known Image Collection Names and their indexes + dictionary keys
print('Collection Names Available:')
for collection in pairGenerator.COLLECTION_KEYS:
    collectionIndex=pairGenerator.COLLECTION_KEYS.index(collection)
    print('Key[{}]:'.format(collectionIndex) +''+collection)

# Choose Image Collection and generate its image pairs
testCollection=pairGenerator.COLLECTION_DICTIONARY['HoChiMinhTraffic_10FPS_1920']
testImagePairs=pairGenerator.generateTestImagePairsFromCollection(testCollection.PATH)
# Run Optical Flow on all image pairs
i=1
for TESTPAIR in testImagePairs[:numImages]:
    print('Calculating Optical Flow For Image [{}/{}]'.format( i,len(testImagePairs) ))
    CalculateOpticalFlow(TESTPAIR)
    i+=1
'''
