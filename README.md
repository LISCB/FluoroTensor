To use FluoroTensor, create a fresh python 3.8.10 virtual environment in an IDE of your choice. We recommend PyCharm.
Download the zipped folder containing FluoroTensor assets, unzip the folder, and copy the files from 'FluoroTensor Project Version 6.6.8 Release' into the folder of the python project created in the IDE.

Run the following commands in the project terminal in the IDE to install the correct versions of the required packages:

pip install tkscrolledframe==1.0.4

pip install openpyxl==3.1.2

pip install numpy==1.26.2

pip install matplotlib==3.4.0

pip install scipy==1.11.4

pip install easygui==0.98.2

pip install scikit-learn==1.3.2

pip install tifffile==2023.9.26

pip install opencv-python==4.8.1.78

pip install tensorflow==2.8.1

pip install pillow==9.5.0


Then run the main program "FluoroTensor.py".
This may produce an error in Tensorflow to do with the protobuf library.
If this is the case, run the additional command to downgrade protobuf:

pip install protobuf==3.20.0

Then try running FluoroTensor again.

The user guide can be found at https://www.spliceselect.org/research/


