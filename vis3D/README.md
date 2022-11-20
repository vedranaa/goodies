## SETUP

- Follow setps from 
https://lab.compute.dtu.dk/patmjen/hcp_tutorials/-/blob/main/HPC_Python_Guide.md I made changes to `init.sh` file, such that it only installs what I need. 

- Install modules. I guess this could be avoided if specifying requirenments in setup.
```
pip install PyQt5
pip install tifffile
pip install Pillow
pip install imagecodecs
```
- From `https://raw.githubusercontent.com/vedranaa/goodies/main/vis3D` get files `vis3D.py` and `setup.py`. For example using `wget` where -N overwrittes if newer, and -O overwrittes into file. (Update, I made `get_code.sh`.)
```
wget path/file.py
wget -N path/file.py
wget path/file.py -O file.py
````

- Editable install


## USE
- Navigate to the folder containing `init.sh` which will set the environment etc.

```
linuxsh -X
source init.sh
```

- Use `vis3D` from the terminal with one of the following
```
vis3D
vis3D path
vis3D file 
```

