from sklearn.decomposition import TruncatedSVD, NMF
import numpy as np
from numpy.linalg import norm
import copy
import matplotlib.pyplot as plt

matrix = np.loadtxt('bars.csv', delimiter=',')

#w fit transform
#N components
#TSNE - je pomala doporucuje se kombinovat pca s tnse

u,s,v = np.linalg.svd(copy.deepcopy(matrix),full_matrices=True)
features = [1,2,5,8,16,32,64]
svd_results = []
nmf_results = []
for num_of_features in features:
    smat = np.zeros((10000, num_of_features), dtype=complex)
    smat[:num_of_features,:num_of_features] = np.diag(s[:num_of_features])
    #umat = np.zeros((10000, 10000), dtype=complex)
    #umat[:,:num_of_features] = u[:,:num_of_features]
    #print(v[:num_of_features,:])
    svd_reconstructed = np.dot(u, np.dot(smat, v[:num_of_features,:]))
    #svd_reconstructed = np.dot(u[:,:num_of_features], np.dot(np.diag(s[:num_of_features]), v[:num_of_features,:])) #for full matrices false
    svd_results.append(norm(svd_reconstructed))

    model = NMF(n_components=num_of_features)
    W = model.fit_transform(copy.deepcopy(matrix))
    H = model.components_
    #print(H)
    nmf_recontructed = np.dot(W,H)
    nmf_results.append(norm(nmf_recontructed))

    print("Num of features {}\n Original matrix norm: {}\nSVD matrix norm: {}\nNMF matrix norm: {}".format(num_of_features,norm(matrix),norm(svd_reconstructed),norm(nmf_recontructed)))


plt.title('Dimension reduction matrix reconstruction')
plt.xlabel('number of features')
plt.ylabel('Frobenius norm value')
plt.plot(features,svd_results,'r',label='svd')
plt.plot(features,nmf_results,'b',label='nmf')
#plt.plot(features,svd_results,'r',features,nmf_results,'b')
plt.legend()
plt.savefig('norms-true')
plt.show()
