import os
'''
    The class in this file is used for ease of life when working with Image collection
        directories. The goal is to make accessing the attributes of an Image Collection
        more intuitive and easier to read for users
'''

class ImageCollection:
    ''' ImageCollection represents an Image Collection Directory as an object.
            It allows us to access its information easily and verbosely
            for the user.
    '''

    def __init__(self, ImageCollectionPath):
        ''' Constructor: Generates all private members about this image collection
        '''
        if( os.path.exists(ImageCollectionPath) ):

            # Path is given
            self.PATH=ImageCollectionPath
            directories=self.PATH.strip('/').split('/')
            # Name is last directory in
            self.NAME=directories[-1]
            self.SOURCE=directories[-2]
            self.SOURCE_PATH=self.PATH.strip(self.NAME)

        else:
            print('Collection path does not exist:',ImageCollectionPath)

# End Image Collection
