import os
import re
from InputCreation.TestImage import TestImage
from InputCreation.TestImagePair import TestImagePair
from InputCreation.ImageCollection import ImageCollection

'''
    The classes in this file are used to cultivate pairs of test images
        for use with the Optical Flow project (PAPTEAM.) The ultimate output
        of these classes will be two absolute paths (stored as strings) to
        corresponding image files

        Strict directory structure:
            'project name'/'source directory'/'image collection'/'image name'
        **This file must stay within the Subdirectories of /'to root'/OpticalFlow/Code/**

        Members:
        ========
            SOURCE_DIRS: a hardcoded list of known source image directories
            PATH_TO_ME: the relative path to this script from the project directory level
            PROJECT_PATH: the absolute path to the project directory level
            COLLECTION_DICTIONARY: dictionary of all image collection objects found in SOURCE_DIRS \
                                        indexed: key=collectionName entry=collectionObject
            COLLECTION_KEYS: list of name of all image collection objects
'''


class TestImagePairGenerator:
    ''' TestImagePairGenerator is a class that is meant to search through the \
            Image Source Directories of the Optical Flow (PAPTEAM) project \
            to create TestImagePairs from all of the source images.
        The expected output of a TestImagePairGenerator will be a long list
            of tuples each of which will contain two absolute paths to corresponding
            test images. the size of and the source for the list will be determined at
            run time by the user

        Source images must be of the form 'JPG' for optical flow operations.
            TestImagePairGenerator will make an effort to ignore non-JPG files
            but the ultimate responsibility falls on the user to protect their
            program from incorrect input images.

        Source Images typically appear in a set of 2 or more images stored within an
            Image Collection folder which are then separated by theme. Image Collections
            are then stored within another source directory which is stored
            at the highest level of the OpticalFlow Project directory structure
            the layout is as follows:
            OpticalFlow/'Source Directory name'/'Image Collection name'/'Image File'


    '''
    def __init__(self):
        # Create relevant paths on start-up
        self.PROJECT_PATH, self.PATH_TO_ME = os.getcwd().split('/Code')
        self.PATH_TO_ME='/Code'+self.PATH_TO_ME
        # Hard coded source folder names
        self.SOURCE_DIRS=('images','images_MPI','images_Video')
        self.COLLECTION_KEYS=list() # Empty list
        self.COLLECTION_DICTIONARY=dict() # Empty Dictionary
        # Image Collections' paths are generated on start-up and not hard coded because
        #   they will probably be prone to updates and shifting around
        self.updateImageCollections()


    # === Operate on Image Collection Directories

    def updateImageCollections(self):
        ''' updateImageCollections: succinct method to update this object's \
                self.COLLECTION_DICTIONARY member.
            This allows the user to refresh the collections list without changing
                the members themselves
        '''
        # Use the generateImageCollections() to generate a list of paths to every
        #   known Image Collection
        self.COLLECTION_DICTIONARY=self.generateImageCollections()


    def generateImageCollections(self):
        ''' generateImageCollections: search all of the known Source Directories \
                (stored in self.SOURCE_DIRS) for all of their children \
                directories and return them as a list of ImageCollection objects.

            These children directories are assumed to be Image Collection folders
                and no verification will be performed by generateImageCollections

            It is assumed that all SOURCE_DIRS are stored directly under the
                Project Directory
        '''
        collectionDictionary=dict() # Empty dictionary
        i=0
        for SOURCEDIR in self.SOURCE_DIRS:
            # Create absolute path to this source dir
            sourcePath=os.path.join(self.PROJECT_PATH,SOURCEDIR)

            # For every directory within source dir
            pathIterator=os.walk(sourcePath)
            for path, imageCollections, images in pathIterator:
                for COLLECTION in imageCollections:
                    # Make Image Collection Object
                    collectionDir=os.path.join(sourcePath,COLLECTION)
                    collectionObject=ImageCollection(collectionDir)
                    # Add Name('key') to key list
                    self.COLLECTION_KEYS.append(collectionObject.NAME)
                    # Add entry to dictionary, key=name entry=object
                    collectionDictionary[collectionObject.NAME]=collectionObject
                for IMAGE in images:
                    i+=1
            # Debug Number of Images found in Collection
            #print('{} total images'.format(i))
        return collectionDictionary


    # === Generate Test Images and Test Image Pairs

    def generateTestImagePairsFromSource(self, sourcePath):
        ''' generateTestImagePairsFromSource: generate TestImagePairs \
                from only one Source Folder (given as an absolute path). \
                Will return list of all Test Image Pairs under Source
                'path to root'/'source dir'
        '''
        TestImagePairs=list()
        if( os.path.exists(sourcePath) ):
            # For every subdirectory of sourcePath
            pathIterator=os.walk(sourcePath)
            for currentPath, imageCollections, files in pathIterator:
                for IMAGECOLLECTION in imageCollections:
                    # List all TestImagePairs within subdirectory
                    collectionPath=os.path.join(sourcePath,IMAGECOLLECTION)
                    collectionImagePairs=self.generateTestImagePairsFromCollection(collectionPath)
                    # for every pair in that list
                    for TESTIMAGEPAIR in collectionImagePairs:
                        # Add to our list
                        TestImagePairs.append(TESTIMAGEPAIR)
        else:
            print('generateTestImagePairsFromSource error:')
            print('Source Path does not exist:',sourcePath)
        return TestImagePairs


    def generateTestImagePairsFromCollectionName(self, collectionName):
        ''' generateTestImagePairsFromCollectionName: Returns all test image Pairs
                found inside the Image Collection Directory 'collectionName'
        '''
        try:
            collectionPath=self.COLLECTION_DICTIONARY[collectionName].PATH
            testImagePairs=self.generateTestImagePairsFromCollection(collectionPath)
            return testImagePairs
        except Exception as e:
            print('Collection Name could not be found with error:',e)


    def generateTestImagePairsFromCollection(self, collectionPath):
        ''' generateTestImagePairsFromCollection: generate TestImagePairs \
                from only one Image Collection (given as an absolute path).
                'path to root'/'source dir'/collectionPath/

            :return TestImagePairs: a list with all of the TestImagePairs that \
                were found within the given image collection directory
            :return type: list of TestImagePair objects
        '''
        TestImagePairs=list() # Empty list
        testImageList=self.generateTestImagesFromCollection(collectionPath)
        for Im_A in testImageList:
            # Guess the next sequential image based on Im_A's filename
            Im_B=Im_A.after()
            # If the sequential image exists
            if( os.path.exists(Im_B.IMAGE_PATH) ):
                imagePair=TestImagePair(Im_A,Im_B)
                TestImagePairs.append(imagePair)
            else:
                pass
        return TestImagePairs

    def generateTestImagesFromCollectionName(self, collectionName):
        ''' generateTestImagesFromCollectionName: Returns all test images
                found inside the Image Collection Directory 'collectionName'
        '''
        try:
            collectionPath=self.COLLECTION_DICTIONARY[collectionName].PATH
            testImages=self.generateTestImagesFromCollection(collectionPath)
            return testImages
        except Exception as e:
            print('Collection Name could not be found with error:',e)

    def generateTestImagesFromCollection(self, collectionPath):
        ''' generateTestImagesFromCollection: generate list of all images \
                within Image Collection Path directory as TestImage objects
        '''
        TestImages=list() # Empty list
        if ( os.path.exists(collectionPath) ):
            # For every subdirectory and file within collectionPath
            pathIterator=os.walk(collectionPath)
            for currentPath, subdirs, imageFiles in pathIterator:
                imageFiles.sort() # sort them in the correct order
                for IMAGEFILE in imageFiles:
                    # Absolute path to this image file
                    imagePath=os.path.join(currentPath,IMAGEFILE)
                    # Create Test Image object from path
                    image=TestImage(imagePath)
                    TestImages.append(image)
        else:
            print('generateTestImagesFromCollection error:')
            print('Collection Path does not exist:',collectionPath)
        return TestImages


# End TestImagePairGenerator

'''
# === Usage
# ======== To run: uncomment these lines and run with `python3 TestImagePairGenerator.py`
#============================These lines are necessary for following demo
generator=TestImagePairGenerator()

imageCollectionIndex=2    # Whatever number you want, within range of COLLECTION_KEYS
collectionName=generator.COLLECTION_KEYS[2]
testCollectionObject=generator.COLLECTION_DICTIONARY[collectionName]
testSourceName=testCollectionObject.SOURCE
testSourcePath=testCollectionObject.SOURCE_PATH
#=================================================================

print('Known Collections:')
for COLLECTIONNAME in generator.COLLECTION_KEYS:
    collection=generator.COLLECTION_DICTIONARY[COLLECTIONNAME]
    print('-'+collection.NAME)

print('\n'+'='*10)
print('Test Collection',testCollectionObject.NAME,'contains images:')
testImages=generator.generateTestImagesFromCollection(testCollectionObject.PATH)
for TESTIMAGE in testImages:
    print('Test Image: '+TESTIMAGE.IMAGE_PARENT+'/'+TESTIMAGE.IMAGE_FILE)

print('\n'+'='*10)
print('Test Image Pairs in collection '+testCollectionObject.NAME+': ')
testImagePairs=generator.generateTestImagePairsFromCollection(testCollectionObject.PATH)
for TESTIMAGEPAIR in testImagePairs:
    print('Collection Pair: '+TESTIMAGEPAIR.asStorageString(delimiterString=' > ',long=False))

print('\n'+'='*10)
print('Test Image Pairs in '+testSourceName+': ')
testImagePairs=generator.generateTestImagePairsFromSource(testSourcePath)
for TESTIMAGEPAIR in testImagePairs:
    print("Source Pair: "+TESTIMAGEPAIR.asStorageString(delimiterString=' > ',long=False))
'''
