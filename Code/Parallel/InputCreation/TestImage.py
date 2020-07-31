import os
import re
from InputCreation.ImageCollection import ImageCollection


class TestImage:
    '''TestImage represents each image file that will be used to test the
            performance of the Optical Flow implementations.

        `This class operates at the path level`

            This class is specifically designed for the image directory
            layout found in the project OpticalFlow:
            'path to root'/OpticalFlow/'source dir'/'image collection'/'image'

            Members:
            ========
                IMAGE_PATH: the exact absolute path to where the image file is found
                                including the filename
                IMAGE_PARENT_PATH: the exact absolute path to the parent directory
                                of this image (like IMAGE_PATH but without filename)
                IMAGE_FILE: the full name of the image's file (including extension)
                IMAGE_NAME: the name of the image without the extension
                IMAGE_EXT: the extension type of the image
                IMAGE_PARENT: the name of the directory the image's file exists in
                IMAGE_SOURCE: the name of the source directory the image's file can
                                be found in (the parent of the parent directory)
                IMAGE_INDEX_INT: input images are typically enumerated 'frameX.jpg'
                                where X is the index. this is the integer version of X
                IMAGE_INDEX_STRING: input images are typically enumerated 'frameX.jpg'
                                where X is the index. this is the string version of X
                                to preserve the number of place values in X (for
                                instance, 'frame00010.jpg')
    '''

    def __init__(self, absPath='./NULL.empty'):
        ''' constructor: instantiates information about image

            There will be no checking that the absPath exists. the .after()
                function relies on being able to make a TestImage object from
                a path that may or may not exist
        :param absPath: exact absolute path to image from root directory
        :type absPath: string
        '''
        self.IMAGE_PATH=absPath
        self.IMAGE_FILE=pathHeight(self.IMAGE_PATH,1)
        self.IMAGE_NAME, self.IMAGE_EXT=self.IMAGE_FILE.split('.')
        self.IMAGE_INDEX_STRING=digitsIn(self.IMAGE_NAME)
        self.IMAGE_INDEX_INT=int(self.IMAGE_INDEX_STRING)
        self.IMAGE_PARENT=pathHeight(self.IMAGE_PATH,2)
        self.IMAGE_PARENT_PATH=absPath.replace(self.IMAGE_FILE,'')
        self.IMAGE_COLLECTION=ImageCollection(self.IMAGE_PARENT_PATH)
        self.IMAGE_SOURCE=pathHeight(self.IMAGE_PATH,3)


    def bio(self):
        ''' bio() creates a biographical string about this TestImage object
                that enumerates all of its members
        '''
        bioString='path: '+self.IMAGE_PATH
        bioString+='\nparent path: '+self.IMAGE_PARENT_PATH
        bioString+='\nName: '+self.IMAGE_NAME
        bioString+='\nIndex: '+str(self.IMAGE_INDEX_STRING)
        bioString+='\nExtension: '+self.IMAGE_EXT
        bioString+='\n Collection: '+self.IMAGE_PARENT
        bioString+='\n Source: '+self.IMAGE_SOURCE
        return bioString


    def after(self):
        ''' after() creates a new TestImage object that would theoretically
                follow this TestImage object, assuming that that next testImage
                actually exists

            for a TestImage created with the file: myImage=.../frame008.jpg,
                after(myImage) will create a TestImage:
                .../frame009.jpg
        '''
        next_file=self.IMAGE_FILE.replace(self.IMAGE_INDEX_STRING,
                                            incrementWithFormat(self.IMAGE_INDEX_STRING))
        return TestImage(os.path.join(self.IMAGE_PARENT_PATH,next_file))


# End TestImage Class



# --- Static Utility Functions
def pathHeight(path, height=2):
    ''' pathHeight() will return the directory index at height 'height'
            for: path=/dir1/dir2/dir3/myfile.txt
            pathHeight(path) gives 'dir3'
            pathHeight(1) gives 'myfile.txt'
            pathHeight(3) gives 'dir2'
            pathHeight(4) gives 'dir1'
            pathHeight(3000) gives 'dir1'
    '''
    pathIndex=path.strip('/').split('/')
    if height<=len(pathIndex):
        return  pathIndex[-height] # Take last directory of path
    else:
        return pathIndex[1]

def digitsIn(string):
    ''' digitsIn() will return all of the numerical digits found within string
            'string' and concatenate them all into a single String
            that is comprised of only numbers

            for myString='my 1st Favorite Number is 345 but I like 2'
            digitsIn(myString) will return '13452'
    '''
    digits=re.sub('\D','',string)   # Replace all non-digits with nothing
    if digits=='':
        return '0'
    else:
        return digits

def incrementWithFormat(string):
    '''incrementWithFormat() will increment the number within a string while
            keeping its width the same as the input

            :param string: a string that is comprised of only numbers. Strings \
                            with non-numbers will lead to unknown behavior
            :type string: string
            This is important for image files where the progression of images is:
                    frame009.jpg > frame010.jpg
                And not:
                    frame009.jpg > frame0010.jpg

            If the incremented number exceeds the alotted number of input place
                Values set by the input string then the spill-over place values
                will be chopped from the output number
            input: output:
            '0'   >   '1'
            '000' >   '001'
            '004' >   '005'
            '009' >   '010'
            '9'   >   '0'
            '99'  >   '00'
            '999' >   '000'
    '''
    inputLength=len(string)
    inputNumber=int(digitsIn(string))
    outputNumber=inputNumber+1
    outputString=str(outputNumber)
    outputLength=len(outputString)
    leadingZeroes=inputLength-outputLength
    if leadingZeroes >= 0:  # Correct length or leading zeroes required
        return ( '0'*leadingZeroes + outputString)
    else:                   # Exceeded number of place values allowed
        return( outputString[-(leadingZeroes):] )
        # we ignore the beginning digits of outputString for as many digits
        #  as we have overflown
    return outputString

# End File
