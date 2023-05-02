'''
Make an html file for image gallery by poining to a folder of images.

# HOW TO USE # 
Read help text for two functions:
    make_gallery,
    fix_images (use of this function is optional).
And look at the example use below.

In fix_images, converting pdf images is done using pdf2image, which is a wrapper 
around poppler. Poppler may be installed using:
conda install -c conda-forge poppler

'''

#%%
import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# MAIN FUNCTIONS
def make_gallery(photos_folder, nr_columns=4, clickable=True, 
                 filename='gallery.html',
                 window_title='Gallery', gallery_title='MY GALLERY', 
                 footer_text='This is my gallery.'):
    
    ''' 
    Make an html file for an folder of images.
    
        photos_folder: string, a path to the folder containing photos. String 
            photos_folder is normally without '/' at the end, and then the htlm 
            document is made in the top folder where photos_folder is placed. If 
            photos_folder has '/' at the end, html document is placed inside the 
            photos folder.
         nr_columns: integer 1, 2, 3, or 4. Number of columns for the image grid.
         clickable: boolean. Whether clicking on the image yields a true-size
            preview of the image. This is useful when images are (natively)
            large and details can't be seen in scaled previews used for the 
            image grid.
        filename: string. Filename with html extension. Don't include the path.
            The path is extracted from photos_folder.
        window_title: string. Shorter text used in browser tabs.
        gallery_title: string. The title of the gallery displayed in html page.
        footer_text: The text placed below the gallery. No footer if empty.

    '''

    #  Get hold of images.
    top_folder, foldername = os.path.split(photos_folder)
    extensions = ['.jpeg', '.jpg', '.png', '.gif']
    files = os.listdir(photos_folder)
    files = sorted(files)
    images = [f for f in files if os.path.splitext(f)[1].lower() in extensions]
    paths = [os.path.join(foldername,  i) for i in images]

    #  Make components.
    pre = make_pre(nr_columns, clickable, window_title, gallery_title)
    photo_grid = make_photo_grid(paths, nr_columns, clickable)
    modal, script = make_click_support(clickable)
    footer = make_footer(footer_text)
    seq1, seq2 = make_seq()

    lines =  pre + photo_grid + modal + footer + seq1 + script + seq2
    
    #  Save the file.
    #  TODO: Check whether file allready exists and abort if yes.
    filepath = os.path.join(top_folder, filename)
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def fix_images(folder_in, folder_out, max_size=2000, 
               from_ext = ['.jpeg', '.jpg', '.png', '.gif'], to_ext='.png', 
               name_as='image'):
    '''
    Processes all images in folder_in and saves in folder_out.

        folder_in: a path to a folder containing images.
        folder_out: a path to a folder where images are to be saved. Needs to be 
            made beforehand (outside this function).
        max_size: int or None. Maximal size of (any) image side.
        from_ext: list of strings or a single string. Extensions considered. 
        to_ext: string or None. Extension used when saving. If None, the 
            original extension is used. 
        name_as: string or None. The root of the image names if images are to be 
            renamed. If None, the original name is used.
        
        TODO Test whether all extensions to and from are supported
    '''

    # Support from only one ext to be considered given as string
    if type(from_ext) is str:
        from_ext = [from_ext]
    
    from_ext = [e.lower() for e in from_ext]

    if '.pdf' in from_ext:
        from pdf2image import convert_from_path

    files = os.listdir(folder_in)
    images = [f for f in files if os.path.splitext(f)[1].lower() in from_ext]
    images = sorted(images)

    if len(images)==0:
        print('No images found!')

    for nr, image in enumerate(images):

        print(f'Processing image nr {nr}: {image}')
        name, ext = os.path.splitext(os.path.split(image)[1])

        if ext == '.pdf':
            # Takes only first page
            I = convert_from_path(os.path.join(folder_in, image))[0]  
        else:
            I = PIL.Image.open(os.path.join(folder_in, image))
        if max_size is not None:
            s = I.size
            I = I.resize([int(max_size * x / max(s)) for x in s])
            print(f'  Resized from {s} to {I.size}')
        if name_as is not None:
            print(f'  Name {name} changed to ', end='')
            name = name_as + str(nr).rjust(len(str(len(images))), '0')
            print(name)
        
        if to_ext is not None:
            if ext != to_ext:
                print(f'  Extension {ext} changed to ', end='')
                ext = to_ext
                print(ext)
        
        I.save(os.path.join(folder_out, name) + ext)



# HELPING FUNCTIONS

def make_photo_grid(paths, nr_columns, clickable):

    # Distribute images in columns.
    # TODO: consider image aspect ratio when distributing to columns.
    min_nr = len(paths)//nr_columns  # at least so many per column
    rest = len(paths) % nr_columns  # to be placed in first x columns
    steps = [min_nr + (i<rest) for i in range(nr_columns)]

    # Make columns.
    lines = ['  <!-- Photo grid -->', 
             '  <div class="w3-row-padding w3-grayscale-min">']
    for i in range(nr_columns):
        k = sum(steps[:i])  # index of first image in this column
        column = make_column(paths[k : k + steps[i]], nr_columns, clickable)
        lines = lines + [''] + column
    lines = lines + ['', '  </div>', '']

    return lines


def make_column(paths, nr_columns, clickable):

    split = {1: 'block', 2:'half', 3: 'third', 4:'quarter'}[nr_columns]
    click = 'onclick="onClick(this)" ' if clickable else ''

    pre = f'   <div class="w3-{split}">'
    seq = '   </div>'
    elements = [f'      <img src="{p}" style="width:100%" {click}alt="">' 
                for p in paths]
    column = [pre] + elements + [seq]

    return column


def make_click_support(clickable):

    modal = ['  <!-- Modal for full size images on click-->',
        '  <div id="modal01" class="w3-modal w3-black" style="padding-top:0" '
                'onclick="this.style.display=\'none\'">',
        '    <span class="w3-button w3-black w3-xlarge '
                'w3-display-topright">×</span>',
        '    <div class="w3-modal-content w3-animate-zoom w3-center '
                'w3-transparent w3-padding-64">',
        '      <img id="img01" class="w3-image">',
        '    </div>',
        '  </div>',
        ''] * clickable
    script = ['<script>',
        '// Modal Image Gallery',
        'function onClick(element) {',
        '  document.getElementById("img01").src = element.src;',
        '  document.getElementById("modal01").style.display = "block";',
        '}',
        '</script>',
        ''] * clickable

    return modal, script


def make_footer(footer_text):

    footer = ['  <!-- Footer -->',
        '  <footer class="w3-container w3-padding-32 w3-light-gray">',
        '    <div class="w3-row-padding">',
        f'      <p>{footer_text}</p>',
        '    </div>',
        '  </footer>',
        ''] * bool(footer_text)

    return footer


def make_pre(nr_columns, clickable, window_title, gallery_title):

    split = {1: 'block', 2:'half', 3: 'third', 4:'quarter'}[nr_columns]
    
    header = ['<!-- Top menu on small screens -->',
        '<header class="w3-container w3-top w3-light-gray w3-padding-16">',
        '  <span class="w3-left  w3-xlarge w3-padding">' + gallery_title + '</span>',
        '</header>',
        ''] * bool(gallery_title)

    pre = (['<!DOCTYPE html>',
        '<html>'] +
        ['<title>' + window_title + '</title>'] * bool(window_title)  + 
        ['<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Raleway">',
        '<style>',
        'body,h1,h2,h3,h4,h5 {font-family: "Raleway", sans-serif}',
        '.w3-' + split + ' img{margin-bottom: 10px' + '; cursor: pointer' * clickable + '}'] +
        ['.w3-' + split + ' img:hover{opacity: 0.6; transition: 0.3s}'] * clickable + 
        ['</style>',
        '',
        '<body class="w3-light-grey">',
        ''] +
        header +
        ['<!-- !PAGE CONTENT! -->',
        '<div class="w3-main w3-content" style="max-width:1600px;margin-top:83px">',
        ''])

    return pre


def make_seq():
    seq1 = ['<!-- End page content -->',
        '</div>',
        '']
    seq2 = ['</body>',
        '</html>',
        '']
    
    return seq1, seq2


#%% Example use
if __name__ == '__main__':
    photos_in = '/Users/VAND/Documents/PROJECTS/goodies/gallery_test/photos_original'
    photos = '/Users/VAND/Documents/PROJECTS/goodies/gallery_test/photos_processed'

    fix_images(photos_in, photos, to_ext='.jpg')
    make_gallery(photos, nr_columns=4, filename='gallery.html')


    # %% Use for 02506, spring 2023
    #photos_in = '/Users/VAND/Documents/TEACHING/02506/02506_2023/posters2023_in'
    #photos = '/Users/VAND/Documents/TEACHING/02506/02506_2023/posters2023/posters2023'
    #fix_images(photos_in, photos, from_ext='.pdf', to_ext='.png', name_as='poster')
    #make_gallery(photos, nr_columns=4, filename='posters2023.html')


    

# %%
