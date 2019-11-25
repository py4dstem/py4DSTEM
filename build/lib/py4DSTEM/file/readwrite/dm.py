"""
A module to load data and meta data from DM3 and DM4 files into python.

On Memory mode:
In a word, support memory mapping. Yes?
The fileDM class supports an "on memory" mode that preloads the file data in memory
and data read operations during data parsing are performed against memory. This
is critical to matain good performance when the file resides in a parallel file
system (PFS) because ithe latency of seek operations PFSs is very high.
"""

import numpy as np
import os
from os.path import basename as osBasename
from os import stat as fileStats
import mmap


class fileDM:

    def __init__(self, filename, verbose=False, on_memory=False):
        '''
        Opens the file and reads its header.

        Accepts:
            filename:   (str) the file path.
            verbose:    (bool) if True, debug information is printed.
            on_memory:  (bool) if True, file data is pre-loaded in memory and all data
                               parsing is performed against memory. Use this mode if the file
                               is in a network based or paralle file system.
        '''

        self.filename = filename

        # necessary declarations, if something fails
        self.fid = None
        self.fidOut = None

        self._on_memory = on_memory

        # check for string
        if not isinstance(filename, str):
            raise TypeError('Filename is supposed to be a string ;D')

        #Add a top level variable to indicate verbose output for debugging
        self.v = verbose

        # try opening the file
        try:
            if not self._on_memory:
                self.fid = open(filename, 'rb')
            if self._on_memory:
                self._buffer_offset = 0
                # Pre-load the file as a memory map that supports operations
                # similar to a file.
                with open(filename, 'rb') as _fid:
                    if os.name == 'nt':
                        self.fid=mmap.mmap(_fid.fileno(), 0,
                                           access=mmap.ACCESS_READ)
                    else:
                        self.fid=mmap.mmap(_fid.fileno(), 0,
                                           prot=mmap.PROT_READ) #, flags=mmap.MAP_PRIVATE)
                    self._buffer_size=fileStats(filename).st_size

        except IOError:
            print('Error reading file: "{}"'.format(filename))
            raise
        except :
            raise

        if not self._validDM():
            #print('Not a valid DM3 or DM4 file: "{}"'.format(filename))
            raise IOError('Can not read file: {}'.format(filename))

        #Lists that will contain information about binary data arrays
        self.xSize = []
        self.ySize = []
        self.zSize = []
        self.zSize2 = [] #only used for 4D datasets in DM4 files
        self.dataType = []
        self.dataSize = []
        self.dataOffset = []
        self.dataShape = [] #1,2,3, or 4. The total number of dimensions in a data set

        #The number of objects found in the DM3 file
        self.numObjects = 0

        #Indicator that a thumbnail exists (tested for later)
        self.thumbnail = False

        self.curGroupLevel = 0 #track how deep we currently are in a group
        self.maxDepth = 64 #maximum number of group levels allowed
        self.curGroupAtLevelX = np.zeros((self.maxDepth,),dtype=np.int8) #track group at current level
        self.curGroupNameAtLevelX = '' #set the name of the root group

        self.curTagAtLevelX = np.zeros((self.maxDepth,),dtype=np.int8) #track tag number at the current level
        self.curTagName = '' #string of the current tag

        #lists that will contain scale information (pixel size)
        self.scale = []
        self.scaleUnit = []
        self.origin = []
        self.dataType = []

        #Temporary variables to keep in case a tag entry shows useful information in an array
        self.scale_temp = 0
        self.origin_temp = 0

        self.outputDic = {}
        self.allTags = {}

    def __del__(self):
        #close the file
        if(not self._on_memory and self.fid):
            if self.v:
                print('Closing input file: {}'.format(self.filename))
            self.fid.close()
        if(self.fidOut):
            if self.v:
                print('Closing tags output file')
            self.fidOut.close()

    def tell(self):
        if self._on_memory:
            return self._buffer_offset
        else:
            return self.fid.tell()

    def fromfile(self, *args, **kwargs):
        """
        Reads data from a file or memory map.

        Calls np.fromfile and np.frombuffer depending on the on_memory mode of
        the load_dmfile instance.

        Args, it supports whatever frombuffer, fromfile support but it requires:
            dtype: np.dtype (object or string) to be read.
            count: number of dtype items to be read.

        Returns: A list of count dtype elements.
        """
        if self._on_memory:
            if "dtype" not in kwargs:
                raise ValueError("In on_memory mode, reads require always a "
                                 "named dtype argument to be specified.")
            if "count" not in kwargs:
                raise ValueError("In on_memory mode, reads require always a "
                                 "named count argument to be specified.")

            dtype=np.dtype(kwargs["dtype"])
            count=int(kwargs["count"])
            kwargs["offset"]=self._buffer_offset
            self._buffer_offset+=int(dtype.itemsize)*count
            return np.frombuffer(*args, **kwargs)
        else:
            # print(*args,**kwargs)
            return np.fromfile(*args, **kwargs)


    def seek(self, fid, offset, from_what=0):
        """
        Positions the reading head for fid.

        fid can be a file or memory map. Follows the same convention as file.seek

        Args:
            fid:       file or memory map.
            offset:    number of bytes to move the head forward (positive value)
                       or backwards (negative value).
            from_what: reference point to use in the head movement. Options:
                         0: beginning of the file (default behavior)
                         1: from the current head position
                         2: from the end of the file.
        """
        if self._on_memory:
            offset=int(offset)
            if from_what==0:
                self._buffer_offset=offset
            elif from_what==1:
                self._buffer_offset+=offset
            elif from_what==2:
                self._buffer_offset=self._buffer_size+offset-1
            else:
                raise ValueError("Unkown from_what value: {}".format(from_what))
            if self._buffer_offset<0:
                raise ValueError("Resulting head position cannot be negative.")
        else:
            return fid.seek(offset, from_what)


    def _validDM(self):
        '''
        Test whether a file is a valid DM3 or DM4 file, and
        verify that it's written in Little Endian format
        '''
        output = True #output will stay == 1 if the file is a true DM4 file

        self.dmType = self.fromfile(self.fid,dtype=np.dtype('>u4'),count=1)[0] #file type: == 3 for DM3 or == 4 for DM4

        if self.v:
            print('validDM: DM file type numer = {}'.format(self.dmType))

        if self.dmType == 3:
            self.specialType = np.dtype('>u4') #uint32
        elif self.dmType == 4:
            self.specialType = np.dtype('>u8') #uint64
        else:
            raise IOError('File is not a valid DM3 or DM4')
            output = False

        self.fileSize = self.fromfile(self.fid,dtype=self.specialType,count=1)[0] #file size: real size - 24 bytes
        self.endianType = self.fromfile(self.fid,dtype=np.dtype('>u4'),count=1)[0] #endian type: 1 == little endian (Intel), 2 == big endian (old powerPC Mac)

        if self.endianType != 1:
            #print('File is not written Little Endian (PC) format and can not be read by this program.')
            raise IOError('File is not written Little Endian (PC) format and can not be read by this program.')
            output = False

        #Test file size for corruption. Note that DM3/DM4 file size is always off by 20/24 bytes from what is written in the header
        osSize = fileStats(self.filename).st_size
        if self.dmType == 3:
            if self.fileSize != osSize-20:
                pass
                #raise IOError('File size on disk ({}) does not match expected file size in header ({}). Invalid file.'.format(osSize, self.fileSize))
                #output = False
                #print('Warning: file size on disk ({}) does not match expected file size in header ({}).'.format(osSize, self.fileSize))
        elif self.dmType == 4:
            if self.fileSize != osSize-24:
                pass
                #raise IOError('File size on disk ({}) does not match expected file size in header ({}). Invalid file.'.format(osSize, self.fileSize))
                #output = False
                #print('Warning: file size on disk ({}) does not match expected file size in header ({}).'.format(osSize, self.fileSize))

        return output

    def parseHeader(self):
        '''Parse the header by simply reading the root tag group. This ensures the file pointer is in the correct place.
        '''
        #skip the bytes read by dmType
        if self.dmType == 3:
            self.seek(self.fid, 12,0)
        elif self.dmType == 4:
            self.seek(self.fid, 16,0)
        #Read the first root tag the same as any other group
        self._readTagGroup()

        #Check for thumbnail
        if(len(self.dataType) > 0): #check that any data set was found
            if( (self.dataType[0] == 23) & (self.dataShape[0] == 2) ):
                self.thumbnail = True
            else:
                self.thumbnail = False
        else: #this file only contains tags (such as a GTG file)
            self.thumbnail = False

    def _readTagGroup(self):
        '''Read a tag group in a DM file
        '''
        self.curGroupLevel += 1
        #Check to see if the maximum group level is reached.
        if self.curGroupLevel > self.maxDepth:
            raise IOError('Maximum tag group depth of {} reached. This file is most likely corrupt.'.format(self.maxDepth))

        self.curGroupAtLevelX[self.curGroupLevel] = self.curGroupAtLevelX[self.curGroupLevel] + 1
        self.curTagAtLevelX[self.curGroupLevel] = 0
        self.fromfile(self.fid,dtype='<i1',count=2) #is open and is sorted?
        nTags = self.fromfile(self.fid,dtype=self.specialType,count=1)[0] #needs to be read as Big Endian (.byteswap() could also work)

        if self.v:
            print('Total number of root tags = {}'.format(nTags))

        #Iterate of the number of tag entries
        oldTotalTag = self.curGroupNameAtLevelX
        for ii in range(0,nTags):
            self._readTagEntry()

        #Go back down a level after reading all entries
        self.curGroupLevel -= 1
        self.curGroupNameAtLevelX = oldTotalTag

    def _readTagEntry(self):
        '''Read one entry in a tag group
        '''
        dataType = self.fromfile(self.fid,dtype=np.dtype('>u1'),count=1)[0]

        #Record tag at this level
        self.curTagAtLevelX[self.curGroupLevel] += 1

        #get the tag
        lenTagLabel = self.fromfile(self.fid,dtype='>u2',count=1)[0]

        if self.v:
            print('_readTagEntry: dataType = {}, lenTagLabel = {}'.format(dataType,lenTagLabel))

        if lenTagLabel > 0:
            tagLabelBinary = self.fromfile(self.fid,dtype='<u1',count=lenTagLabel) #read as binary
            tagLabel = self._bin2str(tagLabelBinary)
            if self.v:
                print('_readTagEntry: tagLabel = {}'.format(tagLabel))
        else:
            tagLabel = str(self.curTagAtLevelX[self.curGroupLevel]) #unlabeled tag.

        #Save the current group name in case this is needed
        oldGroupName = self.curGroupNameAtLevelX

        if dataType == 21:
            #This tag entry contains data
            self.curTagName = tagLabel #save its name
            self._readTagType()
        else:
            #This is a nested tag group
            self.curGroupNameAtLevelX += '.' + tagLabel #add to group names

            #An unknown part of the DM4 tags
            if self.dmType == 4:
                temp1 = self.fromfile(self.fid,dtype=self.specialType,count=1)[0]

            self._readTagGroup()

        self.curGroupNameAtLevelX = oldGroupName

    def _readTagType(self):
        #Need to read 8 bytes before %%%% delimiater. Unknown part of DM4 tag structure
        if self.dmType == 4:
            temp1 = self.fromfile(self.fid,dtype=self.specialType,count=1)[0]

        delim = self.fromfile(self.fid,dtype='<i1',count=4)
        assert((delim == 37).all()) #delim has to be [37,37,37,37] which is %%%% in ASCII.
        if self.v:
            print('_readTagType: should be %%%% = {}'.format(self._bin2str(delim)))

        nInTag = self.fromfile(self.fid,dtype=self.specialType,count=1)[0] #nInTag: unnecessary redundant info

        #Determine the type of the data in the tag
        #specifies data type: int8, uint16, float32, etc.
        encodedType = self.fromfile(self.fid,dtype=self.specialType,count=1)[0] #big endian

        etSize = self._encodedTypeSize(encodedType)

        if etSize > 0:
            #regular data. Read it and store it with the tag name
            if self.v:
                print('regular')
            self._storeTag(self.curTagName, self._readNativeData(encodedType))
        elif encodedType == 18: #string
            if self.v:
                print('string')
            stringSize = self.fromfile(self.fid,dtype='>u4',count=1)[0]
            #strtemp = '' #in case stringSize == 0
            strTempBin = self.fromfile(self.fid,dtype='<u1',count=stringSize) #read as uint8 little endian
            strTemp = self._bin2str(strTempBin)
            self._storeTag(self.curTagName,strTemp)
        elif encodedType == 15: #struct
            #This does not work for field names that are non-zero. This is uncommon
            if self.v:
                print('struct')
            structTypes = self._readStructTypes()
            structs = self._readStructData(structTypes)
            self._storeTag(self.curTagName,structs)
        elif encodedType == 20: #array
            #The array data is not read. It will be read later if needed
            if self.v:
                print('array')
            arrayTypes = self._readArrayTypes() #could be recursive if array contains array(s)
            arrInfo = self._readArrayData(arrayTypes) #only info of the array is read. It is read later if needed
            self._storeTag(self.curTagName,arrInfo)

    def _bin2str(self,bin):
        '''Utility function to convert a numpy array of binary values to a python string
        '''
        return ''.join([chr(item) for item in bin])

    def _encodedTypeSize(self, encodedType):
        '''Return the number of bytes in a data type for the encodings used by DM
        Constants for the different encoded data types used in DM3 files
            SHORT   = 2
            LONG    = 3
            USHORT  = 4
            ULONG   = 5
            FLOAT   = 6
            DOUBLE  = 7
            BOOLEAN = 8
            CHAR    = 9
            OCTET   = 10
            uint64  = 12
            -1 will signal an unlisted type
        '''
        if encodedType == 0:
            return 0
        elif (encodedType == 8) | (encodedType == 9) | (encodedType == 10):
            return 1 #1 byte
        elif (encodedType == 2) | (encodedType == 4):
            return 2 #2 bytes
        elif (encodedType == 3) | (encodedType == 5) | (encodedType == 6):
            return 4 #4 bytes
        elif (encodedType == 7) | (encodedType == 12):
            return 8 #8 bytes
        else:
            return -1

    def _encodedTypeDtype(self,encodedType):
        '''Translate the encodings used by DM to numpy dtypes according to:
            SHORT   = 2
            LONG    = 3
            USHORT  = 4
            ULONG   = 5
            FLOAT   = 6
            DOUBLE  = 7
            BOOLEAN = 8
            CHAR    = 9
            OCTET   = 10
            uint64  = 12
            -1 will signal an unlisted type
        '''
        if encodedType == 2:
            return np.dtype('<i2')
        elif encodedType == 3:
            return np.dtype('<i4')
        elif encodedType == 4:
            return np.dtype('<u2')
        elif encodedType == 5:
            return np.dtype('<u4')
        elif encodedType == 6:
            return np.dtype('<f4')
        elif encodedType == 7:
            return np.dtype('<f8')
        elif encodedType == 8:
            return np.dtype('<u1')
        elif encodedType == 9:
            return np.dtype('<u1')
        elif encodedType == 10:
            return np.dtype('<u1')
        elif encodedType == 12:
            return np.dtype('<u8')
        else:
            return -1

    def _readStructTypes(self):
        '''Analyze the types of data in a struct
        '''
        structNameLength = self.fromfile(self.fid,count=1,dtype=self.specialType)[0] #this is not needed
        nFields = self.fromfile(self.fid,count=1,dtype=self.specialType)[0]
        if self.v:
            print('_readStructTypes: nFields = {}'.format(nFields))

        if(nFields > 100):
            raise RuntimeError('Too many fields in a struct.')

        fieldTypes = np.zeros(nFields)
        for ii in range(0,nFields):
            aa = self.fromfile(self.fid,dtype=self.specialType,count=2) #nameLength, fieldType
            nameLength = aa[0] #not used currently
            fieldTypes[ii] = aa[1]
        return fieldTypes

    def _readStructData(self,structTypes):
        '''Read the data in a struct
        '''
        struct = np.zeros(structTypes.shape[0])
        for ii, encodedType in enumerate(structTypes):
            etSize = self._encodedTypeSize(encodedType) #the size of the data type
            struct[ii] = self._readNativeData(encodedType) #read this type of data
        return struct

    def _readNativeData(self,encodedType):
        '''reads ordinary data types in tags
            SHORT (int16)   = 2;
            LONG (int32)    = 3;
            USHORT (uint16)  = 4;
            ULONG (uint32)   = 5;
            FLOAT (float32)  = 6;
            DOUBLE (float64)  = 7;
            BOOLEAN (bool) = 8;
            CHAR (uint8 character)    = 9;
            OCTET (??)  = 10;
            UINT64 (uint64) = 11;
        '''
        if encodedType == 2:
            val = self.fromfile(self.fid,count=1,dtype='<i2')[0]
        elif encodedType == 3:
            val = self.fromfile(self.fid,count=1,dtype='<i4')[0]
        elif encodedType == 4:
            val = self.fromfile(self.fid,count=1,dtype='<u2')[0]
        elif encodedType == 5:
            val = self.fromfile(self.fid,count=1,dtype='<u4')[0]
        elif encodedType == 6:
            val = self.fromfile(self.fid,count=1,dtype='<f4')[0]
        elif encodedType == 7:
            val = self.fromfile(self.fid,count=1,dtype='<f8')[0]
        elif encodedType == 8: #matlab uchar
            val = self.fromfile(self.fid,count=1,dtype='<u1')[0] #return character or number?
            if self.v:
                print('_readNativeData untested type, val: {}, {}'.format(encodedType,val))
        elif encodedType == 9: #matlab *char
            val = self.fromfile(self.fid,count=1,dtype='<i1')[0] #return character or number?
            if self.v:
                print('_readNativeData untested type, val: {}, {}'.format(encodedType,val))
        elif encodedType == 10: #matlab *char
            val = self.fromfile(self.fid,count=1,dtype='<i1')[0]
            if self.v:
                print('_readNativeData untested type, val: {}, {}'.format(encodedType,val))
        elif encodedType == 11:
            val = self.fromfile(self.fid,count=1,dtype='<u8')[0]
        elif encodedType == 12:
            val = self.fromfile(self.fid,count=1,dtype='<u8')[0] #unknown type, but this works
        else:
            print('_readNativeData unknown data type: {}'.format(encodedType))
            raise

        if self.v:
            print('_readNativeData: encodedType == {} and val = {}'.format(encodedType, val))

        return val
    def _readArrayTypes(self):
        '''Analyze the types of data in an array
        '''
        arrayType = self.fromfile(self.fid,dtype=self.specialType,count=1)[0]

        itemTypes = []

        if arrayType == 15:
            #nested Struct
            itemTypes = self._readStructTypes()
        elif arrayType == 20:
            #Nested array
            itemTypes = _readArrayTypes()
        else:
            itemTypes.append(arrayType)
        if self.v:
            print('_readArrayTypes: itemTypes = {}'.format(itemTypes))
        return itemTypes

    def _readArrayData(self,arrayTypes):
        '''Read information in an array based on the types provided. Binary data is not read at this point.
        '''

        #The number of elements in the array
        arraySize = self.fromfile(self.fid,count=1,dtype=self.specialType)[0]

        if self.v:
            print('_readArrayData: arraySize, arrayTypes = {}, {}'.format(arraySize,arrayTypes))

        #Everything used to calculate the bufSize is not needed anymore. THis can be removed after testing
        itemSize = 0
        for encodedType in arrayTypes:
            if self.v:
                print('_readArrayData: encodedType = {}'.format(encodedType))
            etSize = self._encodedTypeSize(encodedType)
            itemSize += etSize

        bufSize = arraySize * itemSize
        bufSize = bufSize.astype('<u8') #change to an integer

        if self.v:
            print('_readArrayData: arraySize, itemSize = {}, {}'.format(arraySize, itemSize))

        if self.curTagName == 'Data':
            #This is a binary array. Save its location to read later if needed
            self._storeTag(self.curTagName + '.arraySize', bufSize)
            self._storeTag(self.curTagName + '.arrayOffset', self.tell())
            self._storeTag(self.curTagName + '.arrayType', encodedType)
            self.seek(self.fid, bufSize.astype('<u8'),1) #advance the pointer by bufsize from current position
            arrOut = 'Data unread. Encoded type = {}'.format(encodedType)
        elif bufSize < 1e3: #set an upper limit on the size of array that will be read in as a string
            #treat as a string
            for encodedType in arrayTypes:
                stringData = self.fromfile(self.fid,count=arraySize,dtype=self._encodedTypeDtype(encodedType))
                arrOut = self._bin2str(stringData)

            #This is the old way to read this in. Its not really correct though.
            #stringData = self.bin2str(self.fromfile(self.fid,count=bufSize,dtype='<u1'))
            #arrOut = stringData.replace('\x00','') #remove all spaces from the string data

            #Catch useful tags for images and spectra (nm, eV, etc.)
            fullTagName = self.curGroupNameAtLevelX + '.' + self.curTagName
            if((fullTagName.find('Dimension') > -1) & (fullTagName.find('Units') > -1) ):# & (self.numObjects > 0)):
                self.scale.append(self.scale_temp)
                self.scaleUnit.append(arrOut)
                self.origin.append(self.origin_temp)
        else:
            self._storeTag(self.curTagName + '.arraySize', bufSize)
            self._storeTag(self.curTagName + '.arrayOffset', self.tell())
            self._storeTag(self.curTagName + '.arrayType', encodedType)
            self.seek(self.fid, bufSize.astype('<u8'),1) #advance the pointer by bufsize from current position
            arrOut = 'Array unread. Encoded type = {}'.format(encodedType)

        return arrOut

    def _storeTag(self,curTagName,curTagValue):
        '''Builds the full tag name and key/value pair as text. Also calls another
        function to catch useful tags and values. Also saves all tags in a dictionary.
        '''
        #Build the full tag name (key) and add the tag value
        if self.v:
            print('_storeTag: curTagName, curTagValue = {}, {}'.format(curTagName,curTagValue))
        totalTag = self.curGroupNameAtLevelX + '.' + '{}'.format(curTagName) #+ '= {}'.format(curTagValue)

        self._catchUsefulTags(totalTag,curTagName,curTagValue)

        self.allTags[totalTag] = curTagValue #this needs to be done better.

        return(totalTag)

    def _catchUsefulTags(self,totalTag,curTagName,curTagValue):
        '''Find interesting keys and keep their values for later. This is separate from _storeTag
        so that it is easy to find and modify.
        '''

        #Save that a useful object has been found
        if totalTag.find('ImageData.Calibrations.Dimension.1.Scale')>-1:
            self.numObjects += 1 #data is contained in this file

        if curTagName.find('Data.arraySize')>-1:
            self.dataSize.append(curTagValue)
        elif curTagName.find('Data.arrayOffset') >-1:
            self.dataOffset.append(curTagValue)
        elif curTagName.find('DataType')>-1:
            self.dataType.append(curTagValue)
        elif totalTag.find('Dimensions.1')>-1:
            self.xSize.append(curTagValue)
            self.ySize.append(1)
            self.zSize.append(1)
            self.zSize2.append(1)
            self.dataShape.append(1) # indicate as at least 1D data
        elif totalTag.find('Dimensions.2')>-1:
            self.ySize[-1] = curTagValue #OR self.ysize[self.numObjects] = self.curTagValue
            self.dataShape[-1] = 2 # indicate as at least 2D data
        elif totalTag.find('Dimensions.3')>-1:
            self.zSize[-1] = curTagValue
            self.dataShape[-1] = 3 # indicate as at least 3D data
        elif totalTag.find('Dimensions.4')>-1:
            self.zSize2[-1] = curTagValue
            self.dataShape[-1] = 4 # indicate as at least 3D data
        elif (totalTag.find('Dimension.')>-1) & (totalTag.find('.Scale')>-1):
            self.scale_temp = curTagValue
        elif (totalTag.find('Dimension.')>-1) & (totalTag.find('.Origin')>-1):
            self.origin_temp = curTagValue
        else:
            pass

    def writeTags(self):
        fnameOutPrefix = self.filename.split('.dm3')[0]
        try:
            #open a text file to write out the tags
            with open(fnameOutPrefix+'_tags.txt','w') as fidOut:
                for nn in sorted(self.allTags):
                    try:
                        oo = '{} = {}'.format(nn,str(self.allTags[nn]))
                        fidOut.write(oo)
                    except:
                        fidOut.write('{} = dm.py error'.format(nn))
                    fidOut.write('\n')
            fidOut.close() #this might not be necessary
        except NameError:
            print("Issue opening tags output file.")
            raise
        except:
            raise

    def _checkIndex(self, i):
        '''Check index i for sanity, otherwise raise Exception.

        Parameters:
            i (int):    Index.

        '''

        # check type
        if not isinstance(i, int):
            raise TypeError('index supposed to be integer')

        # check whether in range
        if i < 0 or i > self.numObjects:
            raise IndexError('Index out of range, trying to access element {} of {} valid elements'.format(i+1, self.head['ValidNumberElements']))

        return

    def _DM2NPDataType(self, dd):
        '''Convert the DM data type value into a numpy dtype
        '''
        if dd == 6:
            return np.uint8
        elif dd == 10:
            return np.uint16
        elif dd == 11:
            return np.uint32
        elif dd == 9:
            return np.int8
        elif dd == 1:
            return np.int16
        elif dd == 7:
            return np.int32
        elif dd == 2:
            return np.float32
        elif dd == 12:
            return np.float64
        #elif dd == 14: #this is supposed to be bit1 in matlab, but Im not sure what that translates to in numpy
        #    return np.uint8 #bit1 ??
        elif dd == 3:
            return np.complex64
        elif dd == 13:
            return np.complex128
        elif dd == 23:
            raise IOError('RGB data type is not supported yet.')
            #return np.uint8
        else:
            print("4321")
            raise IOError('Unsupported binary data type during conversion to numpy dtype. DM dataType == {}'.format(dd))

    def getDataset(self, index):
        '''
        Retrieve a dataset from the DM file.

        Note: Most DM3 and DM4 files contain a small "thumbnail" as the first dataset written as
        RGB data. This function ignores that dataset if it exists. To retrieve the thumbnail use
        the getThumbnail() function
        '''
        #The first dataset is usually a thumbnail. Test for this and skip the thumbnail automatically
        if self.numObjects == 1:
            ii = index
        else:
            ii = index + 1

        #Check that the dataset exists.
        try:
            self._checkIndex(ii)
        except:
            raise

        self.seek(self.fid, self.dataOffset[ii],0) #Seek to start of dataset from beginning of the file

        outputDict = {}

        outputDict['filename'] = osBasename(self.filename)

        #Parse the dataset to see what type it is (image, image series, spectra, etc.)
        if self.xSize[ii] > 0:
            pixelCount = int(self.xSize[ii])*self.ySize[ii]*self.zSize[ii]*self.zSize2[ii]
            jj = 0 #counter to determine where the first scale value starts
            for nn in self.dataShape[0:ii]:
                    jj += nn #sum up all number of dimensions for previous datasets
            #if self.dataType == 23: #RGB image(s)
            #    temp = self.fromfile(self.fid,count=pixelCount,dtype=np.uint8).reshape(self.ysize[ii],self.xsize[ii])
            if self.zSize[ii] == 1: #2D data
                outputDict['data'] = self.fromfile(self.fid,count=pixelCount,dtype=self._DM2NPDataType(self.dataType[ii])).reshape((self.ySize[ii],self.xSize[ii]))
                outputDict['pixelUnit'] = self.scaleUnit[jj:jj+self.dataShape[ii]][::-1] #need to reverse the order to match the C-ordering of the data
                outputDict['pixelSize'] = self.scale[jj:jj+self.dataShape[ii]][::-1]
                outputDict['pixelOrigin'] = self.origin[jj:jj+self.dataShape[ii]][::-1]
            elif self.zSize2[ii] > 1: #4D data
                # outputDict['data'] =  np.memmap(self.fid,dtype=self._DM2NPDataType(self.dataType[ii]) ## Delete?
                outputDict['data'] = np.memmap(self.fid,dtype= '>u2',
                                               mode='r',shape=(self.zSize2[ii],self.zSize[ii],
                                               self.ySize[ii],self.xSize[ii]))
                # outputDict['data'] = self.fromfile(self.fid,count=pixelCount,
                #                     dtype=self._DM2NPDataType(self.dataType[ii])
                #                     ).reshape((self.zSize2[ii],self.zSize[ii],
                #                     self.ySize[ii],self.xSize[ii]))
                outputDict['pixelUnit'] = self.scaleUnit[jj:jj+self.dataShape[ii]][::-1] #need to reverse the order to match the C-ordering of the data
                outputDict['pixelSize'] = self.scale[jj:jj+self.dataShape[ii]][::-1]
                outputDict['pixelOrigin'] = self.origin[jj:jj+self.dataShape[ii]][::-1]
            else: #3D array
                outputDict['data'] = self.fromfile(self.fid,count=pixelCount,dtype=self._DM2NPDataType(self.dataType[ii])).reshape((self.zSize[ii],self.ySize[ii],self.xSize[ii]))
                outputDict['pixelUnit'] = self.scaleUnit[jj:jj+self.dataShape[ii]][::-1] #need to reverse the order to match the C-ordering of the data
                outputDict['pixelSize'] = self.scale[jj:jj+self.dataShape[ii]][::-1]
                outputDict['pixelOrigin'] = self.origin[jj:jj+self.dataShape[ii]][::-1]

        return outputDict

    def _readRGB(self,xSizeRGB,ySizeRGB):
        '''Read in a uint8 type array with [Red,green,blue,alpha] channels.
        '''
        return self.fromfile(self.fid,count=xSizeRGB*ySizeRGB*4,dtype='<u1').reshape(xSizeRGB,ySizeRGB,4)

    def getThumbnail(self):
        '''Read the thumbnail saved as the first dataset in the DM file as an RGB array
        Unsure if this is correct.
        '''
        self.seek(self.fid, self.dataOffset[0],0)
        return self._readRGB(self.ySize[0],self.xSize[0])

def dmReader(fName, dSetNum=0, verbose=False):
    '''
    Simple function to parse the file and read the requested dataset
    '''
    f1 = fileDM(fName,verbose)      #initialize a fileDM class instance, opening the file
    f1.parseHeader()                #parse the header
    im1 = f1.getDataset(dSetNum)    #get the requested dataset (first by default)
    del f1                          #delete the class and close the file
    return im1                      #return the dataset and metadata as a dictionary
