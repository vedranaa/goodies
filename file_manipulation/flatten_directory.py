# -*- coding: utf-8 -*-
"""
Flattening a tree-like folder structure. I used it for collecting photographs
from tree-like folder structure into a single folder, ready to 
vand, 2022
"""

import glob
import os
import shutil


def get_extensions(description):
    if description == 'all':
        extensions = True
    elif description == 'photo':
        extensions = ['jpeg', 'jpg', 'png', 'gif', 'bmp',
                    'avi', 'qt', 'mov', 'mp4', 'm4p', 'mpg', 'mpeg']
    elif description == 'image':
        extensions = ['jpeg', 'jpg', 'png', 'gif', 'tiff', 'pdf', 'eps']
    else:
        extensions = description
    return extensions


def copy_flattened(root, destination, copy_all='all', ask_first=False):
    '''  Copy files from directory tree to destination directory.

    Parameters
    ----------
    root : string
        Root of the directory tree. Tested only on absolute paths.
    destination : string
        Destination folder. Tested only on absolute paths.
    copy_all : string or list of strings
        List of extensions to copy, e.g. ['txt', 'png', 'pdf']. 
        If string, 'all' copies all files, 'photo' copies photo and video 
        files, 'image' copies common image files.
    ask_first : Bool, optional
        Whether to ask for confirmation before copying. The default is False.
        !!! SPYDER 5.1 HAS PROBLEMS WITH FUNCTION input. DON'T USE ASK_FIRST
        !!! IF RUNNING FROM AFFECTED SPYDER.

    Returns
    -------
    copy : list of tupples
        Files (to be) copied.
    skip : list of stringe
        Files (to be) skipped.
    processed : list of strings.
        Folders processed.

    '''
    
    # TODO check whether destination exists, make new or abort if not
    
    copy, skip, processed = flatten(root, copy_all)
    print(f'Processed {len(processed)} folders.')
    print(f'About to copy {len(copy)} files.')
    print(f'Skipping {len(skip)} files')
    
    if ask_first and not (input('Proceed? ').lower() in 'yes'):
        print('Aborting')
    else:
        print('Proceeding with copy.')
        for c in copy:
            #print(c[1])
            dst = os.path.join(destination, c[1])
            shutil.copyfile(c[0], dst)
    
    print('Done!')
    return copy, skip, processed


def flatten(foldername, copy_all='all', prefix='', copy=[], skip=[], processed=[]):

    processed.append(foldername)    
    files = glob.glob(foldername + '/*')
    L = len(foldername) + 1  # + 1 due to '/' 
    extensions = get_extensions(copy_all)
    
    for f in files:
        if os.path.isdir(f):
            copy, skip, processed = flatten(f, copy_all, prefix + '_' + f[L:])
        else:
            fn, ext = os.path.splitext(f)
            # First part evaluates to false if string or list.
            if  (extensions==True) or (ext[1:].lower() in extensions):  
                n = prefix + '_' + f[L:]
                copy.append((f, n[1:]))
            else:
                skip.append(f)
               
    return copy, skip, processed


#p = '/Users/VAND/Documents/PROJECTS/goodies/goodies/test_root'
#d = '/Users/VAND/Documents/PROJECTS/goodies/goodies/test_des'
#copy, skip, processed = flatten(p)
#print(copy)
#copy, skip, processed = copy_flattened(p, d)


# photos = '/Volumes/Toshiba1/PICTURES/PHOTOS'
# a = [f[len(photos)+1:] for f in (sorted(glob.glob(photos+'/*')))]
# with open('list.txt', 'w') as f:
#     for l in a:
#         f.write(l + '\n')

#%%

photos = '/Volumes/Toshiba1/PICTURES/PHOTOS/2005'
flat = '/Users/vand/Documents/_PHOTOS_FLATTENED/2005'
copy, skip, processed = flatten(photos, copy_all='photo')

#%%
check = [s for s in skip if s[-3:]!='.db']
[print(c) for c in check]

#%%
copy, skip, processed = copy_flattened(photos, flat, copy_all='photo')
