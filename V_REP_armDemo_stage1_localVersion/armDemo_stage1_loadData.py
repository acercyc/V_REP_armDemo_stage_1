import os
import numpy as np
from numpy import array as npa
import matplotlib.image as mpimg
from scipy import interpolate
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter

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
def imgPreprocess(img):
    # cut to the box part
    img_f = img[yRange[0]:yRange[1], xRange[0]:xRange[1], 0:3]
    return img_f


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
def genOutputMat(c, img_f, gaussFilter=0, scaleByHigh=False):
    outmats = []
    for i in range(c.shape[0]):
        x = np.floor(m2p_x(c[i, 0])).astype(np.int)
        y = np.floor(m2p_y(c[i, 1])).astype(np.int)
        outmat = np.zeros_like(img_f)[:, :, 0]
        outmat[y, x] = 1

        if gaussFilter > 0:
            outmat = gaussian_filter(outmat, gaussFilter)
            outmat = outmat / np.max(outmat)

        if scaleByHigh:
            outmat = outmat * c[i, 2]

        outmats.append(outmat)

    outmats = np.stack(outmats, 2)
    outmats = np.max(outmats, 2)
    outmats = outmats / np.max(outmats)

    return outmats


# ============================================================================ #
#                                construct data                                #
# ============================================================================ #
def loadData(sampleRange, dataPath, runID, gaussFilter=0, scaleByHigh=False):
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
        data_y.append(genOutputMat(samples, img_f, gaussFilter, scaleByHigh))
    data_x = npa(data_x)
    data_y = np.expand_dims(npa(data_y), 4)
    return data_x, data_y


def loadData_withSplit(sampleRange, dataPath, runID, test_size=0.3):
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
    d = train_test_split(data_x, data_y, test_size=test_size, shuffle=True)
    d_train = [d[0], d[2]]
    d_test = [d[1], d[3]]

    return d_train, d_test
