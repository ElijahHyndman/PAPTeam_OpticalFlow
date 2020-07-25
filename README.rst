+++++++++++++++++++++++++++++++++++++++
Parallel Processing Team: Optical Flow
+++++++++++++++++++++++++++++++++++++++

.. contents::
    :depth: 3


Authorship Notice
==================

.. topic:: Authors

    -*The bulk of the code here was not written by me and I make no claims to on their authorship. My name has been placed on scripts I have written, and made no indication on scripts I have edited*
    
    * **Original Authors**: Ce Liu and Deepak Pathak 
    
    * Link to `Ce Liu's Work <https://people.csail.mit.edu/celiu/OpticalFlow/>`_
    
    * Link to `Deepak Pathak's Repository <https://github.com/pathak22/pyflow>`_
    
    
Structure
==========

This repository (now) contains both the Code and Input Images to calculate Optical Flow. 

Input Images
^^^^^^^^^^^^

Images are assumed to be kept in an ``image folder`` which contains sub-folders (termed ``Image Collections``), each with several test ``.png`` images named ``frame###.png``.

example: ``OpticalFlow/images/HoChiMinhTrafficMovement/frame0000098.png``

    * The image folder should be named ``images`` or ``images_video`` (if you are running the *Ho Chi Minh Traffic* test sequences)

    * The image folder **must** be located next to the Code directory. i.e.: ``Home/OpticalFlow/Code`` would imply the image collections are located in ``Home/OpticalFlow/images``

    * The ``Image Collection Folders`` can be named whatever you want, its name will later be used to determine its output folders' names
    
 Not following this Input Image Structure may lead to some unexpected behavior!

Code
^^^^
    * All code is located within 
        ``Code`` 
    * All C++ Code is located within 
        ``Code/Parallel/pyflow/src`` 
    * All Python code for locating and constructing Test Images (from ``images`` folder) is located within 
        ``Code/Parallel/pyflow/InputCreation`` 
    * The **actual program** is located within 
        ``Code/Parallel/pyflow`` 
        
        These scripts are\:
        
        * TestSuite.py
        * OpticalFlowCalculation.py
    * The Cython code for linking the C++ code to the python is located within 
        ``Code/Parallel/pyflow``
        
        Its input files consist of\:
        
        * pyflow.pyx
        * coarse2Fine.pxd
        * setup.py
        
        Its output files consist of\:
        
        * pyflow.cpp
        * pyflow.cpython-36m-x86_64-linux-gnu.so
        * ``build/`` temp.linux-x86_64-3.6
        
        (Yes, Cython files are not pretty, but it turned out to be super handy!)

Output
^^^^^^

Output from running the ``TestSuite.py`` script will be located in the ``Code/Parallel/pyflow/output`` folder. 

    * Output sub-folders will be organized by *Collection Name*
    * Images will be stored in separate folders separated based on run time parameters
    * All timing for an Image Collection will be located at ``output/"Collection Name"/timing``



Usage
======

Actually calculating the optical flow now is quite easy

You can write any Test Runs you want to execute within the ``Parallel/pyflow/TestSuite.py`` file using the ``TestRun()`` function

All console commands will be executed from the ``Parallel/pyflow`` level

There are a few dependencies that need to be downloaded for Cython, Python, and C++ that may cause your first few attempts to fail. Read the error messages to figure out what to download, it'll work eventually! :)

    * To run the TestSuite.py file, simply run\: ::

        python3 TestSuite.py
        
If you wish to make edits to the C++ code or pyflow.pyx Cython stuff, you will need to recompile.

    * To Compile, run\: ::

        python3 setup.py build_ext --inplace

:author: Elijah Hyndman
:Date: July 2020
