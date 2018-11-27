"""
Title:A classify machine for the flowers iris
Author: Lekun ZHUANG
github:https://github.com/LekunZHUANG
"""
from numpy import *
import matplotlib.pyplot as plt
import operator

"""
KNN algorithm
Input: inX: the Vector we are going to test
       dataSet:the inpu of the test set
       labels: the labels of the result in the test set
       K: the number of neighbour we choose
Output: sortedClassCount[0][0]:the most frequency appeared class in the k nearest neighbour
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet      #获取检验点与所有点的差矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                  #差矩阵平方在行方向上求和
    distances = sqDistances**0.5                         #计算距离
    sortedDistIndicies = distances.argsort()             #获取排序后的下标数组
    classCount ={}
    for i in range(k):                                   #将k个距离最小点的标签,出现次数写入字典
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] =classCount.get(voteIlabel, 0) + 1
    #字典排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#step1:prepare the data
#read the data from the file and put them in the matrice
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)          #get the numbers of lines
    returnMat = zeros((numberOfLines-1, 4))     #creat the same format matrix witl all 0
    classLabelVector = []
    index = 0
    #for i in range(len(arrayOLines)):
    #if the first line of file is attribute, activate this line
    for i in range(1, len(arrayOLines)):
        line = arrayOLines[i]
        line = line.strip()                        #remove '\n' on the right side
        listFromLine = line.split(',')            #split on a ',' into list of substrings
        returnMat[index, :] = listFromLine[0:4]       #Feature Matrix
        classLabelVector.append(str(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#step2:analysis the data
#make some scatter plot of the data
def scatterplot():
    axislabel = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    dataMat, dataLabels = file2matrix('irisdata.csv')
    dataLabels = array(dataLabels)
    idx_1 = where(dataLabels == 'Iris-setosa')
    idx_2 = where(dataLabels == 'Iris-versicolor')
    idx_3 = where(dataLabels == 'Iris-virginica')
    i = 0
    while i < 3:
        plt.figure()
        p1 = plt.scatter(dataMat[idx_1, i], dataMat[idx_1, i+1], marker='*', color='r')
        p2 = plt.scatter(dataMat[idx_2, i], dataMat[idx_2, i+1], marker='+', color='b')
        p3 = plt.scatter(dataMat[idx_3, i], dataMat[idx_3, i+1], marker='o', color='g')
        plt.xlabel(axislabel[i])
        plt.ylabel(axislabel[i+1])
        plt.legend((p1, p2, p3), ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
        #k = int(i/2+1)
        #plt.savefig('graphs/figure' + str(k))
        i = i+2
    plt.show()

#step3:Data processing
#Normalize the data int the number between 0 and 1
def autoNorm(dataSet):
    minVals = dataSet.min(0)            #the (0) means we get the minVals of each column
    maxVals = dataSet.max(0)
    ranges = maxVals -minVals            #pay attention that ranges is not 'range'
    normDataSet = zeros(shape(dataSet))  #initialize the normal dataSet with all 0
    m = dataSet.shape[0]
    #newValue = (oldValue-minValue)/(maxValue-minValue)
    normDataSet =dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#step4:Test the algorithm
#test the algorithem with Knn and get the error rate
def classTest():
    hoRatio = 0.2                            #in the file we take 20% of it as a test group
    DataMat, DataLabels = file2matrix('irisdata.csv')
    normMat, ranges, minVals = autoNorm(DataMat)
    m = int(normMat.shape[0]/3)                   #divide in three
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    #in the file we take 80% of it as a train group
    trainnormMat = vstack((normMat[numTestVecs:m],
                           normMat[numTestVecs+50:m*2],
                           normMat[numTestVecs+100:m*3]))
    trainLabels = concatenate((DataLabels[numTestVecs:m],
                               DataLabels[numTestVecs+50:m*2],
                               DataLabels[numTestVecs+100:m*3]))
    for j in range(0, 3):
        for i in range(numTestVecs):
            classifierResult = classify0(normMat[j*50+i], trainnormMat, trainLabels, 3)
            print('the classify came back with:%s, the real answer is %s'
                  % (classifierResult, DataLabels[j*50+i]))
            if (classifierResult != DataLabels[j*50+i]):
                errorCount = errorCount + 1.0
    errorRate = errorCount/float(numTestVecs*3)
    print('the total error rate is :' + str(errorRate))
    return errorRate

#step5:use the algorithm
#use the algorithm to predict the class of a new iris
def classifyIris():
    print('Please enter the features of the Iris you see!')
    sepal_length = float(input("The sepal length:"))
    sepal_width = float(input("The sepal width:"))
    petal_length = float(input("The petal length:"))
    petal_width = float(input("The petal width:"))
    dataMat, dataLabels = file2matrix('irisdata.csv')
    normMat , ranges, minVals = autoNorm(dataMat)
    inArr = array([sepal_length, sepal_width, petal_length, petal_width])
    inArr = (inArr - minVals)/ranges
    classifierResult = classify0(inArr, normMat, dataLabels, 3)      #k=3
    print("The iris is probably:" + classifierResult)

#additional:To know the best percentage I take from data as traning set
#How do I divide the whole data set into training set and test set has minimum error rate?
def ClassTest(testSetRatio):
    hoRatio = testSetRatio
    DataMat, DataLabels = file2matrix('irisdata.csv')
    normMat, ranges, minVals = autoNorm(DataMat)
    m = int(normMat.shape[0] / 3)  # divide in three
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # in the file we take 80% of it as a train group
    trainnormMat = vstack((normMat[numTestVecs:m],
                           normMat[numTestVecs + 50:m * 2],
                           normMat[numTestVecs + 100:m * 3]))
    trainLabels = concatenate((DataLabels[numTestVecs:m],
                               DataLabels[numTestVecs + 50:m * 2],
                               DataLabels[numTestVecs + 100:m * 3]))
    for j in range(0, 3):
        for i in range(numTestVecs):
            classifierResult = classify0(normMat[j * 50 + i], trainnormMat, trainLabels, 3)
            #print('the classify came back with:%s, the real answer is %s'
            #      % (classifierResult, DataLabels[j * 50 + i]))
            if (classifierResult != DataLabels[j * 50 + i]):
                errorCount = errorCount + 1.0
    errorRate = errorCount / float(numTestVecs * 3)
    #print('the total error rate is :' + str(errorRate))
    return errorRate

def ErrorRatePlot():
    testSetRatio = linspace(0.02, 0.98, 49)
    ErrorRate = []
    for i in range(len(testSetRatio)):
        ErrorRate.append(ClassTest(testSetRatio[i]))
    plt.figure("kNN algorithem error rate line")
    plt.xlabel("The percentage of test set in the whole data set")
    plt.ylabel("The error rate of prediction")
    plt.plot(testSetRatio, ErrorRate, color='red')
    #plt.savefig('graphs/ErrorRate')
    plt.show()

if __name__ == '__main__':
    #step2:analysis the data
    scatterplot()
    #step4:test the algorithm
    classTest()
    #step5:use the algorithm
    classifyIris()
    # #additional: error rate line
    ErrorRatePlot()
