#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:39:45 2020
@author: vand@dtu.dk
"""

import sys
import os
import shutil
import glob

def image_cleanup(texfile, destination = 'cleaned', flatten = True, imfolder = ''):
    """Utility for image cleanup on texfile. Files used in texfile, and the 
    texfile, will be copied to destionation folder. If flatten is True, all 
    files will be collected in a single directory, and tex file will be
    copied and modified accordingly.
    Arguments:
        texfile: a path+filename of texfile to be scanned
        destination: a path+foldername of the folder to be created and populated
        flatten=True: a flag indicating flat image directory or directory tree
        imfolder='': a foldername (no path) for images in destination 

    Examples of use:
        image_cleanup('texfolder/filename.tex', 'texfolder_cleaned_flat)                                           
        image_cleanup('texfolder/filename.tex', 'texfolder_cleaned_flatter', imfolder = 'figures')
        image_cleanup('texfolder/filename.tex', 'texfolder_cleaned_tree', flatten = False)
    
    Assumes:
        * Only one \includegraphics per line
        * \includegraphics and image filename are on the same line
        * Only one tex, i.e. no \include files
        * Commenting only by % as a first non-space character in a line
        * Destination folder does not exist
        * no \graphicspath
        * no files in strange places outside branch where tex is 
        
    Author: vand@dtu.dk, 2020
    """
    # TODO: support for \graphicspath
    # TODO: support for \begin{comment} and \end{comment}
    # TODO: support for continuing search in next line if imfiletext not found 
    # CHECK how it works if images are not in subfolders but in parent folder or even elswhere
    # TODO: when called from another folder, adjust paths 
    # DONE: support for line which is commented out
    # DONE: support for flatten dirstructure (dir/bla/im.png -> dir_bla_im.png), also copy+modify texfile!


    def find_imfiletext(line):
        """ Finding image filename text in the line. Many things can go wrong! """
        start = line.index('\includegraphics')+len('\includegraphics')
        if line[start]=='[': # content of options needs to be ignored
            start += line[start:].index(']')+1 # will throw error if ']' is in next line
        if line[start]=='{': # this should be true    
            end = start + line[start:].index('}')    
            return line[start+1:end], line[:start+1], line[end:]
        else:
            return None           
       
    def new_file(file):
        """new file depending on whether to flatten"""
        if flatten:
            newfile = os.path.join(os.path.join(destination, imfolder), file.replace('/', '_'))
        else:
            newfile = os.path.join(destination,file)
        return newfile  
     
    def copy_image(imfiletext, rooth):
        """ Copying image file. Many things can go wrong. """    
        if flatten: # needs to return new text for tex file
            if imfolder:  # this is latex, not os, therfore not using os.path.join
                new_imfiletext = imfolder + '/' + imfiletext.replace('/', '_')
            else:
                new_imfiletext = imfiletext.replace('/', '_')
        else: # making directory if needed
            os.makedirs(os.path.join(destination, os.path.split(imfiletext)[0]) , exist_ok = True)  
            new_imfiletext = None
        old_file = os.path.join(rooth, imfiletext)
        if os.path.exists(old_file): # copying file if extension is given
            shutil.copyfile(old_file, new_file(imfiletext))
            print(f'.+..Copying "{imfiletext}"')
        else: # need to figure out which extension to use
            files = glob.glob(old_file + '.*')
            if len(files)==0:
                print(f'!!..WARNING: Did not find files matching "{imfiletext}".')
                return None
            if len(files)>2:
                print(f'!!..WARNING: Found multiple files matching "{imfiletext}". Copying all!')
            for f in files:
                shutil.copy(f, new_file(imfiletext + os.path.splitext(f)[1]))
                print(f'.+..Copying "{f}"')  
        return new_imfiletext
    

    # MAIN FUNCTIONALITY STARTS HERE   
    try:
        os.mkdir(destination)
        print('STARTING')
        print(f'.+..Creating folder "{destination}"')
    except:
        print(f'ERROR: Seems that destination directory "{destination}" already exists. Delete it before cleaning.')
        return # I insist on empty (new) destination
    
    texroot, texfilename = os.path.split(texfile)
    
    if flatten: # also need to copy the tex file
        newf = open(os.path.join(destination, texfilename), "w")
        os.makedirs(os.path.join(destination, imfolder) , exist_ok = True)        
    
    # reading texfile line by line
    with open(texfile, 'r') as f:
        for line in f:
            new_imfiletext = None # a flag for successfull copy of image file
            if '\includegraphics' in line:   
                try:
                    imfiletext, before, after = find_imfiletext(line) # finding image filename in line                       
                except:
                    imfiletext = None
    
                if line.lstrip()[0]=='%': # a line starting with the comment
                    if imfiletext:                  
                        print(f'....Ignoring image file "{imfiletext}" in a commented line')
                    else:
                        print(f'....Ignoring a commented line "{line}"')
                else: # line which is not commented
                    if imfiletext:
                        try:
                            new_imfiletext = copy_image(imfiletext, texroot)
                        except:
                            print(f'!!..WARNING: Something went wrong copying image file "{imfiletext}"')
                    else:
                        print(f'!!.. WARNING: Something went wrong when processing line "{line}"')
            
            if flatten: # needs to copy tex file line-by-line
                if new_imfiletext:
                    newf.write(before + new_imfiletext + after)
                else:
                    newf.write(line)
    if flatten:
        newf.close()
    else:      
        shutil.copy(texfile,os.path.join(destination,texfilename))
          
# only default functionality from CL       
if len(sys.argv)<2:
    print('Usage: "./image_cleanup.py path/folder/file.txt".')
    print('This creates directory "path/cleaned".')
    print('Note, only default functionality from CL.')
if len(sys.argv)==2:
    image_cleanup(sys.argv[1], os.path.join(os.path.split(os.path.split(sys.argv[1])[0])[0],'cleaned'))
if len(sys.argv)==3:
    image_cleanup(sys.argv[1], sys.argv[2])
    
    