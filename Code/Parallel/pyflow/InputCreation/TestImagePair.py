import os
from InputCreation.TestImage import TestImage

class TestImagePair:
    ''' TestImagePair holds two TestImage objects-- TestImage member 'BEFORE' as \
            the logical first TestImage, and TestImage member 'AFTER' as the logical \
            second TestImage

        `This class operates at the Test Image level`

        Optical Flow usually operates on just two images at a time, one image at
                time 't' and another image at 't+1' so the goal of TestImagePairs
                is to make accessing these sequential images easy. The idea is that
                one Optical Flow Operation will only interact with one TestImagePair
                at a time.
    '''
    storageDelimiter='#separator#' # Used to separate TestImage paths when storing in a text file

    def __init__(self, before, after):
        ''' constructor: Takes two input TestImage objects and stores them
                as a TestImagePair.

            :param before: The TestImage that is logically before the AFTER TestImage
            :type before: TestImage
            :param after: The TestImage that is logically after the BEFORE TestImage
            :type after: TestImage
        '''
        self.BEFORE=before
        self.AFTER=after


    def asStorageString(self, delimiterString=storageDelimiter, long=True):
        ''' asStorageString() generates a string to best store the contents of \
                the TestImagePair.
            TestImagePairs can be generated from only two TestImages and TestImages
                can be generated from just an absolute path to the image file so we
                will store TestImagePairs as two absolute paths in one line separated
                by a delimiter
        '''
        #storageString=self.BEFORE.IMAGE_PATH+delimiter+self.AFTER.IMAGE_PATH
        if long:
            storageString=self.BEFORE.IMAGE_PATH+delimiterString+self.AFTER.IMAGE_PATH
        else:
            storageString=self.BEFORE.IMAGE_PARENT+'/'+self.BEFORE.IMAGE_NAME+ \
                            delimiterString+ \
                            self.AFTER.IMAGE_PARENT+'/'+self.AFTER.IMAGE_NAME
        return storageString
# End TestImagePair
