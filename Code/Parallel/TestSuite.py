# Author: Elijah Hyndman Date: July 2020
from OpticalFlowCalculation import *
try: # import our own local made modules
    from InputCreation.TestImagePairGenerator import TestImagePairGenerator as TestImagePairGenerator
    from InputCreation.TestImagePair import TestImagePair as TestImagePair
    from InputCreation.TestImage import TestImage as TestImage
    from InputCreation.ImageCollection import ImageCollection as ImageCollection
except Exception as e:
     print('*'*3+'Input Creation Modules were not imported: {}'.format(e))

import time

# === Variables to make work easier and customizeable
CollectionLevels={
    'Light':'HoChiMinhTraffic_10FPS_320',
    'Medium':'HoChiMinhTraffic_10FPS_800',
    'Heavy':'HoChiMinhTraffic_10FPS_1920'
}
FinalLevels=('HoChiMinhTraffic_10FPS_240','HoChiMinhTraffic_10FPS_480','HoChiMinhTraffic_10FPS_960','HoChiMinhTraffic_10FPS_1920')
CollectionProgressions={
    'Light':(CollectionLevels['Light'],),
    'Default':(CollectionLevels['Medium'],),
    'Heavy':(CollectionLevels['Heavy'],),
    'Full':(CollectionLevels['Light'],CollectionLevels['Medium'],CollectionLevels['Heavy']),
    'Final':FinalLevels
}


# === Test Run Function
def TestRun(Progression='Default',ImagesPerCollection=3,pyramidLevels=3,threadProgression=(1,2,3,4),Repetitions=1,outputSuffix=''):
    ''' TestRun will complete a full Test run on the Test Image Pairs found in the Image Collections within CollectionProgressions
            :Progression: refers to a specific progression of Image Collection Directories, found at the top of the file
            :ImagesPerCollection: Allows user to only run a few images from each Collection ('-1' to do All)
            :pyramidLevels: Specifies the height of the LK Pyramid used to calculate optical flow
            :threadProgression: for each image, OF calculations are repeated with different number of threads each time, specified
                                    by this progression. each entry adds another OF calculation per image
            :Repetitions: Allows the user to run the entire TestRun multiple Times if specified
    '''
    # Debug at beginning of Test Run
    print('='*100)
    print('\tProgression:',Progression,'=',CollectionProgressions[Progression])
    print('\t',ImagesPerCollection,'Images each:',pyramidLevels,'Level Pyramids: Thread Progression of ',threadProgression,': Repeated',Repetitions,'\n')

    ImagePairGenerator=TestImagePairGenerator()


    # Total Progress bar: Calculate the total number of images that we will work with
    totalImages=len(threadProgression)*Repetitions
    #   Images per collection may be:
    #       - Specified number of Images or,
    #       - Entire arbitrarily large directory
    if ImagesPerCollection>=1:
        numberOfCollections=len(CollectionProgressions[Progression])
        totalImages*=ImagesPerCollection*numberOfCollections
    else:
        # This is a roundabout way to count the number of images within each test image collection directory
        collectionImages=0
        for eachCollection in CollectionProgressions[Progression]:
            collectionImages+=len(ImagePairGenerator.generateTestImagePairsFromCollectionName(eachCollection))
        totalImages*=collectionImages


    # Run the entire test 'Repetitions' times over
    imageIndex=1
    # For number of Repetitions
    for iteration in range(Repetitions):
        print('='*20,'Iteration',iteration+1,':\n')
        # For every image collection named
        for COLLECTION in CollectionProgressions[Progression]:
            ImagePairs=ImagePairGenerator.generateTestImagePairsFromCollectionName(COLLECTION)
            # For each Number of Threads
            for numCores in threadProgression:
                print('+'*15,numCores,'threads','+'*15,'\n')
                # For each Image Pair
                for IMAGEPAIR in ImagePairs[:ImagesPerCollection]:

                    # === DEBUG: Print Calculation Header
                    progressString='Image ['+str(imageIndex)+'/'+str(totalImages)+']'
                    print('['+IMAGEPAIR.asStorageString(' -> ',long=False)+']', '='*10 , progressString, '='*10)

                    # === Perform calculation
                    CalculateOpticalFlow(IMAGEPAIR,pyramidLevels,numCores,outputSuffix)

                    # === DEBUG: Print Calculation footer
                    print( '=' * 60, '\n')
                    imageIndex+=1
    # == End Test Run


# === Running Tests
begin=time.perf_counter()
TestRun(Progression='Final',ImagesPerCollection=10,pyramidLevels=6,threadProgression=(2,4,8,16,24),Repetitions=1,outputSuffix='_parallel')
print('Program Execution time: {:.2f}'.format(time.perf_counter()-begin))
