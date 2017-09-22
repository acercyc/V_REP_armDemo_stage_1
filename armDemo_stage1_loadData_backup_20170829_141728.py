import os
import numpy as np
import matplotlib.image as mpimg
from scipy import interpolate
from sklearn.model_selection import train_test_split

# ============================================================================ #
#                                  Parameters                                  #
# ============================================================================ #
xRange = (17, 239)
yRange = (73, 183)
xBound = (0, 1)
yBound = (0, 0.5)

# ============================================================================ #
#                               Mapping function                               #
# ============================================================================ #
m2p_x = interpolate.interp1d([xBound[0], xBound[1]], [0, xRange[1] - xRange[0] - 1])
m2p_y = interpolate.interp1d([yBound[1], yBound[0]], [0, yRange[1] - yRange[0] - 1])

p2m_x = interpolate.interp1d([0, xRange[1] - xRange[0] - 1], [xBound[0], xBound[1]])
p2m_y = interpolate.interp1d([0, yRange[1] - yRange[0] - 1], [yBound[1], yBound[0]])


# ============================================================================ #
#                                Load data table                               #
# ============================================================================ #
def loadDataTable(dataPath, runID):
    dataTablePath = os.path.join(dataPath, runID + '.txt')
    dataTable = np.loadtxt(dataTablePath)

    # filter out "out of boundary" data
    xf = np.logical_and(dataTable[:, 0] > xBound[0], dataTable[:, 0] < xBound[1])
    yf = np.logical_and(dataTable[:, 1] > yBound[0], dataTable[:, 1] < yBound[1])
    dataTable = dataTable[np.logical_and(xf, yf), :]

    return dataTable


# ============================================================================ #
#                                  image load                                  #
# ============================================================================ #
def loadImage(iImg, dataPath, runID):
    imgPath = os.path.join(dataPath, runID)
    imgPath = os.path.join(imgPath, '{:d}.png'.format(iImg))
    img = mpimg.imread(imgPath)
    # xy somehow is rotated
    img_f = img[yRange[0]:yRange[1], xRange[0]:xRange[1], 0:3]
    # plt.imshow(img_f)
    # plt.show()
    return img_f


# ============================================================================ #
#                                  Output mat                                  #
# ============================================================================ #
def genOutputMat(c, img_f):
    outmat = np.zeros_like(img_f)[:, :, 0]
    for i in range(c.shape[0]):
        x = np.floor(m2p_x(c[i, 0])).astype(np.int)
        y = np.floor(m2p_y(c[i, 1])).astype(np.int)
        outmat[y, x] = 1
    return outmat


# ============================================================================ #
#                                construct data                                #
# ============================================================================ #
def loadData(sampleRange, dataPath, runID):
    # sampleRange = range(0, 10)

    dataTable = loadDataTable(dataPath, runID)
    data_x = []
    data_y = []
    for i in sampleRange:
        iSample = dataTable[:, 3] == i
        if not np.any(iSample):
            continue
        samples = dataTable[iSample, 0:3]
        img_f = loadImage(i, dataPath, runID)
        data_x.append(img_f)
        data_y.append(genOutputMat(samples, img_f))
    d = train_test_split(data_x, data_y, test_size=0.1, shuffle=True)
    d_train = [d[0], d[2]]
    d_test = [d[1], d[3]]

    return d_train, d_test
