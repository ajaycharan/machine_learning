import numpy as np
from PIL import Image
#import scipy.misc
import os,sys


class segmentation:

	def __init__(self, K=3, inputImageFilename="dog.jpg", outputImageFilename="dog-segmented.jpg"):
		self.K = K
		self.inputImageFilename = inputImageFilename
		self.outputImageFilename = outputImageFilename
		self.img = None
		self.featureMat = None
		self.centroids = None
		self.labels = None
		self.numClusteredPts = None
		self.mean = None
		self.dev = None

	def getFeatureMat(self, img):
		im = img.load()
		count = 0
		width, height = img.size
		featureMat = np.zeros((width*height, 5))
		#print featureMat
		for i in range(width):
			for j in range(height):
				featureMat[count, :] = [im[i,j][0], im[i,j][1], im[i,j][2], i, j]
				count += 1
		#print featureMat
		return(featureMat)

	def normalize(self, featureMat):
		self.mean = np.mean(featureMat, axis=0)
		self.dev = np.std(featureMat, axis=0)
		featureMat = np.subtract(featureMat, self.mean)
		featureMat = np.divide(featureMat, self.dev)
		return(featureMat)

	def KmeansUpdate(self, featureMat):
		n, d = featureMat.shape
		labels_old = np.zeros((1,n))
		
		#print comparison
		#print '----------------'
		#trueArr = (np.zeros(n) == np.zeros(n))
		#print 'TRUE ARR', trueArr
		while(np.array_equal(labels_old, self.labels) != True):
			labels_old = self.labels
			dist = np.zeros((n,self.K))
			
			for j in range(self.K):
				count = 0
				Pts = np.zeros((1,5))
				for i in range(n):
					if (labels_old[i] == j):
						Pts += featureMat[i,:]
						count +=1
				self.centroids[j, :] = np.divide(Pts, count)
				#self.centroids[j] = np.mean(Pts)
				#print 'count', count
				dist[:, j] = np.linalg.norm((featureMat-self.centroids[j, :]),axis=1)
			self.labels = np.argmin(dist, axis=1)
			#print np.array_equal(labels_old, self.labels)
			#comparison = (labels_old == self.labels)
			#print comparison
			#print '----------------'
		
		#print 'stopped upDATWE'
		#print labels_old
		#print '============'
		#print self.labels
		return(self.centroids)

	def firstIteration(self, featureMat):
		n, d = featureMat.shape
		indices = np.arange(n)
		np.random.shuffle(indices)
		#print 'INDEX', indices
		tmpMat = featureMat[indices]
		self.centroids = np.zeros((self.K, 5))
		#self.labels = np.zeros()
		self.centroids = tmpMat[0:self.K, :]
		#print 'centroids', self.centroids
		dist = np.zeros((n,self.K))
		for i in range(self.K):
			dist[:, i] = np.linalg.norm((featureMat-self.centroids[i, :]),axis=1)
		#print 'dist', dist
		self.labels = np.argmin(dist, axis=1)
		#print 'labels', self.labels

	def segment(self, inputImageFilename, outputImageFilename):
		img = Image.open(inputImageFilename)
		width, height = img.size
		#print height, width
		#img.show()
		featureMat = self.getFeatureMat(img)
		featureMat = self.normalize(featureMat)
		#print 'featureMat', featureMat
		self.firstIteration(featureMat)
		self.centroids = self.KmeansUpdate(featureMat)
		#print 'first done'
		'''
		self.centroids *= self.std
		self.centroids += np.around(self.mean)
		'''
		self.centroids = np.int_(np.rint(np.add(np.multiply(self.centroids, self.dev), self.mean)))
		#print self.centroids
		rgbMat = self.centroids[:, 0:3]
		outMat = rgbMat[self.labels]
		outIm = np.reshape(outMat, (width, height, 3))
		#print 'outMat', outMat
		'''
		outIm = np.zeros((height, width, 3), dtype=np.uint8)
		
		count = 0 #width*height
		for i in range(height):
			for j in range(width):
				#count -= 1
				r, g, b = outMat[j+i*height][0], outMat[j+i*height][1], outMat[j+i*height][2]
				print r, g, b
				outIm[i][j][0] = r
				outIm[i][j][1] = g
				outIm[i][j][2] = b
				#featureMat[count, :] = [im[i,j][0], im[i,j][1], im[i,j][2], i, j]
				count += 1
		
		#outMatt = np.reshape(rgbMat[self.labels, :], (width, height, 3))
		'''
		outImg = Image.fromarray(np.uint8(outIm))
		outImg = outImg.transpose(Image.ROTATE_270)
		outImg = outImg.transpose(Image.FLIP_LEFT_RIGHT)
		outImg.save(self.outputImageFilename)
		#outImg.show()
		#scipy.misc.imshow(outIm)

if __name__ == "__main__":

	K = int(sys.argv[1])
	inputImageFilename = sys.argv[2]
	outputImageFilename = sys.argv[3]
	#inputImageFilename = "dog.jpg"
	#outputImageFilename = "dog-segmented.jpg"
	segmented = segmentation( K = K, inputImageFilename = inputImageFilename, outputImageFilename = outputImageFilename)
	segmented.segment(inputImageFilename, outputImageFilename)
