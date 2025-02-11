# DR_WareHouse & WareHouse_1 
These two are the inventory of codes for calculating properties for molecular clouds, including


- excitation temperature
- optical depth
- column density for 12CO, 13CO and C18O
- distance measurement
- mass


and other methods for handling the fits file.

# QuickCheck

QuickCheck is the GUI software for quickly checking the data by simply clicking the button.

The advantages of this software are as follows:
- Handle at most three 3D datacubes at the same time
- Perform a quick check of data's spectra for any region
- Create P-V diagrams by coordinates or any two points in the map
- Match with at most two catalogues
- Make adjustable contours for graphs
- Save graphs for scientific use
- Easy to add new functionalities...

## Guide to QuickCheck

The QuickCheck has four Frames, each of them is for a special use.

### The First and Second Frames

- **The first Frame** contains three sections:
  - The file loading and selecting section
    > Load at most three three-dimensional datacubes and one two-dimensional datacube
  - The first graph section
    > Plot the moment 0 map of the selected file
  - The coordinate input section
    > Input user-determined coordinates to create P-V diagrams (in Frame 3)

- **The second Frame** contains two sections:
  - The second graph section
    > Plot the averaged spectra based on the user-selected region in the first graph
  - The multi-functional section
    > The first five spinboxes are for selection of user-determined velocity ranges to make integration maps
    >
    > The button "Plot Background" is for plotting the graph in Frame 3 (based on the region selected in graph 1)
  - Other buttons' functions will be present in the demostration

<img width="1427" alt="截屏2025-02-11 07 33 01" src="https://github.com/user-attachments/assets/5d260dea-df92-4c80-8fed-3dad720b8320" />


- **The Third Frame** contains two sections:
  - The third graph section
    > Plot the "Background" based on the user-selected region in the first graph.
  - The multi-functional section
    > Load up to two catalogues and mark them on the third graph.
    >
    > Change contour levels for both the P-V diagram in Frame 4 and the red-shift and blue-shift integration maps based on the selected range in Frame 2.
    >
    > Create a P-V trajectory (using user-determined coordinates or user-drawn points).

- **The Fourth Frame** contains one section:
  - Plot the P-V diagram based on the P-V trajectory
    > Use the mouse left button to select a desired region and press "Enter" to plot the averaged spectra for this region and mark the corresponding position in Frame 3.

### Here is a demonstration

<img width="888" alt="截屏2025-02-11 11 53 36" src="https://github.com/user-attachments/assets/a844cd91-8683-4f57-9014-d31e25029e74" />


**Use Mouse (left) to mark a box to zoom-in, and then press "Enter" to create the averaged spectra for this selected region**


**Every Frame has equipped with zoom-in (Mouse left to select box) and zoom-out (Mouse right to return to the previous layer) function**

<img width="999" alt="截屏2025-02-11 10 18 35" src="https://github.com/user-attachments/assets/41ead878-1ceb-4c9e-aae9-892b9712e1d4" />

<img width="959" alt="截屏2025-02-11 11 33 39" src="https://github.com/user-attachments/assets/af173bf2-3563-4ae1-a3cf-aad6304dd154" />




