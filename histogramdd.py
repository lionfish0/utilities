import numpy as np

def histogramdd(sample, bins):
    #sample : array_like
    #The data to be histogrammed. It must be an (N,D) array.
    #
    #bins : list of numpy arrays, specifying the boundaries of the bins
    #(number of bins M1...MD)
    #
    #temporary memory usage, for res:
    #(M1+2)x(M2+2)x...x(MD+2)

    #res is the resulting histogram (temporarily it's M+1 x M+1)
    res = np.zeros([1+np.array(b).shape[0] for b in bins],dtype=np.int16)
    #indices is the location in the histogram of each data point.
    indices = np.zeros([len(bins),sample.shape[0]],dtype=np.int16)
    
    #iterate through the dimensions (store the location of each point
    #in each dimension in indices). Takes NxD memory
    for i,b in enumerate(bins):
        #iterate through the bins, to get the index of each value.
        #(to emulate the behaviour of the original histogramdd
        #we make the last bin inclusive)
        for boundary in b[:-1]: 
            indices[i,:]+=(sample[:,i]>=boundary)
        indices[i,:]+=(sample[:,i]>b[-1])
        
    #(this line doesn't work if more than one item is in a cell:
    #res[indices[0,:],indices[1,:]]+=1)
    #so having to loop...
    #loop through each column (data point) of indices. Increment the appropriate
    #cell in the historgram tensor.
    for index in indices.T:
        idx = [slice(v,v+1) for v in index]
        res[idx]+=1
    
    #strip the first and last rows/cols from the tensor
    #as these are either below the first boundary, or above the last.
    idx = [slice(1,-1)] * (res.ndim)
    res=res[idx]
    return res
    
