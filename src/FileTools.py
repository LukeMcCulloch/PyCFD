# -*- coding: utf-8 -*-
"""
Created on Tue Jan 07 09:49:19 2014

@author: Luke.McCulloch
"""

import sys
import os
import platform
if platform.system() == 'Windows':
    slash = '\\'
else:
    slash = '/'

def fileTools():
    read = 'r'
    write = 'w'
    access_mode = [read,write]
    buffering = -1
    return access_mode, buffering


def GetFile(rootFldr, defaultFileName ):
    """Find and open a file"""
    if not defaultFileName:
        File = raw_input("Please enter Desired filename: %s"%defaultFileName + chr(8)*4)
        #if not File:
        #    File = defaultFileName 
    else:
        File = defaultFileName
    for root, dirs, files in os.walk(rootFldr): 
        for fCntr in files: 
            if fCntr == File: 
                rootFldr = root 
    return rootFldr, File
    

def GetLines(directory,filename):
    """ 
        inputs  :   string(directory)
                    string(filename)
                    
        outputs :   list[lines of the file]
    """
    access_mode, buffering = fileTools()    
    rootFldr = directory
    defaultFileName = filename
    FilePath, FileName = GetFile(rootFldr, defaultFileName )
    fileHandle = open(FilePath+slash+FileName, access_mode[0], buffering)
    lines = fileHandle.readlines()
    fileHandle.close()
    return lines
    


def GetLineByLine(directory,filename):
    """ 
        inputs  :   string(directory)
                    string(filename)
                    
        outputs :   list[lines of the file]
    """
    access_mode, buffering = fileTools()    
    rootFldr = directory
    defaultFileName = filename
    FilePath, FileName = GetFile(rootFldr, defaultFileName )
    print('opening mesh case:')
    print(FilePath+slash+FileName)
    fileHandle = open(FilePath+slash+FileName, access_mode[0], buffering)
    return fileHandle

def WriteLines(directory, filename, lines):
    access_mode, buffering = fileTools()    
    rootFldr = directory
    defaultFileName = filename
    FilePath, FileName = GetFile(rootFldr, defaultFileName )
    fileHandle = open(FilePath+slash+FileName, access_mode[1], buffering)
    for line in lines:
        fileHandle.write(line)
    fileHandle.close()
    return
    
def main():
    #directory = 'C:\Program Files (x86)\Bentley\Engineering\SACS 5.6 V8i SS3'
    #filename =   'lnhinp.demo10'
    #directory = 'C:\Users\luke.mcculloch\Documents\My Codes\Bentley\MOSES-SACS\TOW_testing\Transport\Tow\test11'
    #directory = 'C:\test11'
    directory = r'C:\Users\luke.mcculloch\Documents\My Codes\Bentley\MOSES-SACS\SACSBargetoMOSESbarge'
    filename =   'log00001.txt'
    testlines = GetLines(directory,filename)
    for line in testlines:
        print( line )
    return testlines
    
if __name__=='__main__':
   lines =  main()
