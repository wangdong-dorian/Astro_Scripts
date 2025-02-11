# DR_WareHoouse & WareHouse_1 
These two are the inventory of codes for calculating properties for molecular clouds, including


- excitation temperature
- optical depth
- column density for 12CO, 13CO and C18O
- distance measurement
- mass


and other tools for handling the fits datacube.

# QuickCheck
QuickCheck is the GUI software for quickly check the data by simply click the button.


The advantages of this software are as follows :
- handle at most three 3D datacube at the same time
- do the quick check of data's specta for any region
- create P-V diagrams by coordinates or any two points in the map
- match with at most catalogues
- make adjustable contours for graphs
- save graphs for scientific use
- easy to add new functionalities...

## Guide to QuickCheck
The QuickCheck has four Frames, each of them is for a special use.


### The first and second Frames
- The first Frame contains three sections:
  - the file loading and selecting section
    > load at most 3 three-dimensional datacube and 1 two-dimensional datacube
  - the first graph section
    > draw the moment 0 map of select file
  - the coordinate input section
    > user input coordinates to create P-V diagram (in Frame3)
- The second Frame contains two sections:
  - the second graph section
    > draw the averaged spectra based on the user-selected region in the first graph
  - the multi-functional section
    > the first five spinbox is for selection of user-determined velocity range to make integration map
    
    > the button "Plot Background" is for plotting the graph in Frame3 (based on the region selected in graph1)

<img width="1427" alt="截屏2025-02-11 07 33 01" src="https://github.com/user-attachments/assets/5d260dea-df92-4c80-8fed-3dad720b8320" />


> Here is a demostration


<img width="951" alt="截屏2025-02-11 09 40 30" src="https://github.com/user-attachments/assets/9665d8cf-bd54-4edd-bead-513eab98d6e0" />


> Use Mouse (left) to mark a box to zoom-in, and then press "Enter" to create the averaged spectra for this selected region



**Every Panel has equipped with zoom-in (Mouse left to select box) and zoom-out (Mouse right to return to the previous layer) function**

- The third and forth Frames

<img width="999" alt="截屏2025-02-11 10 18 35" src="https://github.com/user-attachments/assets/41ead878-1ceb-4c9e-aae9-892b9712e1d4" />




