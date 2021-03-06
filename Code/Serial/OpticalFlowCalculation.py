# Author: Elijah Hyndman 2020

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
import matplotlib.pyplot as plt
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

def CalculateOpticalFlow(imagePair, pyramidLevels, optionalOutputSuffix=''):
    ''' CalculateOpticalFlow calculates the optical flow for two TestImage objects, \
            image_A and image_B

            :param imagePair: TestImagePair that will be used to calculate
                                Optical Flow
            :param pyramidLevels: an integer height for the LK compression pyramid
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
    timingDictionary, u, v, im2W = pyflow.coarse2fine_flow(im1, im2, pyramidLevels)
    # === Output Vectors to array
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    debugTimingDiagnostics(timingDictionary)

    # === All of the output Storing
    generateOutput(imagePair, pyramidLevels, flow, imDimensions, timingDictionary, optionalOutputSuffix)



def debugTimingDiagnostics(timingDictionary):
    # Copy values to a dictionary that we can edit freely
    timing=timingDictionary.copy()
    print('Total Time for Image:',timingDictionary['Total C++ Execution'])
    print(timing)



def generateOutput(imagePair, nLevels, flow, imDimensions, timingDictionary, optionalOutputSuffix):
    ''' Generate all output relevant to Optical Flow

        :param imagePair: TestImagePair object that was used to create \
                            Optical Flow output
        :param flow: np array of (u,v) coordinates for vectors
        :param im1: np array created from one of the JPG RGB input images
    '''

    # === Folder Names for Path
    # All output will be stored in ./output
    # This output is stored in ./output/ImageCollectionName
    # images and timing are stored in separate respective folders
    outputPath=os.path.join(os.getcwd(),'output')
    Collection_outputFolderName=imagePair.BEFORE.IMAGE_PARENT+optionalOutputSuffix
    Collection_outputPath=os.path.join(outputPath,Collection_outputFolderName)

    Image_outputPath=os.path.join(Collection_outputPath, ('images_P{}'.format(nLevels)) )
    Timing_outputPath=os.path.join(Collection_outputPath, ('timing') )

    # Create said Directories if they don't exist
    os.makedirs(Collection_outputPath,exist_ok=True)
    os.makedirs(Image_outputPath,exist_ok=True)
    os.makedirs(Timing_outputPath,exist_ok=True)

    # === Files
    # Image File Name (based on first image of pair)
    Frame_Base=('frame{}'.format(imagePair.BEFORE.IMAGE_INDEX_STRING))
    Image_OutputExtension='.jpg'
    Image_OutputFile=Frame_Base+Image_OutputExtension
    OUTPUT_IMAGEFILE_PATH=os.path.join(Image_outputPath,Image_OutputFile)

    # Timing File
    if (0):
        # Timing File Name (based on pyramid height, thread number)
        Timing_OutputFile=Collection_outputFolderName+'_P{}.txt'.format(nLevels)
        OUTPUT_TIMINGFILE_PATH=os.path.join(Timing_outputPath,Timing_OutputFile)
    else:
        #Timing_OutputFile=Collection_outputFolderName+'.txt'
        Timing_OutputFile='UniversalTiming'+'.txt'
        OUTPUT_TIMINGFILE_PATH=os.path.join(outputPath,Timing_OutputFile)
    ImageSize=imagePair.BEFORE.IMAGE_PARENT.split('_')[-1]
    frameDeclaration=ImageSize+' pixels '+str(nLevels)+' pyramid\t'

    # === Create the final output files
    if(0):
        generateOutputFlowImageFile(OUTPUT_IMAGEFILE_PATH,flow,imDimensions)
    generateOutputTimingFile(outputFilePath=OUTPUT_TIMINGFILE_PATH,timingDictionary=timingDictionary,lineStarter=frameDeclaration)



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



def generateOutputTimingFile(outputFilePath, timingDictionary, lineStarter=''):
    # We will write the header first, if the file does not exist yet
    print('printing output',outputFilePath)
    writeHeader= not os.path.exists(outputFilePath)
    f = open(outputFilePath,'a')
    delimiter='\t'

    if( writeHeader ):
        HeaderString=delimiter.join( timingDictionary.keys() )
        # Write path in header
        f.write(outputFilePath+'\n')
        # Write Header Names
        if(not lineStarter==''):
            f.write('Image')
            f.write(delimiter)
        f.write(HeaderString+'\n')

    # Join all of the values in the timingDictionary, delimit with delimiter string
    if(not lineStarter==''):
        f.write(lineStarter)
        f.write(delimiter)
    timingString=delimiter.join( timingDictionary.values() )
    f.write(timingString+'\n')
    f.close()

def generateOutputGraphs():
    pass
