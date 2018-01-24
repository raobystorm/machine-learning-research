#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits>
#include <boost/python.hpp>

using namespace std;

extern float **Rank1Count(float **vecs,float **gallery,int G,int N,int F,int K,int gallerySize);

extern int compare_IndexedFloats (const void *a, const void *b);

struct IndexedFloat {
  float val;
  int index;
  int source;
};

#define CLUSTER 1
#define GALLERY 2

BOOST_PYTHON_MODULE(erclustering)
{
    using namespace boost::python;
    def("Rank1Count", &Rank1Count);
}

// This version of Rank1 Count uses an "external" gallery.
// That is, it uses a gallery that is not part of the set of images to 
// be clustered.

// This function takes as input an FxN (F is usually 4096) matrix corresponding 
// to N CNN feature vectors to be clustered.
//
// It returns an (float **) pointing to an NxN array of "rank-1 counts" for each pair 
// of images (A,B). The higher the rank-1 counts, the more likely a pair (A,B) is
// to be "joined". Using a threshold, one can convert this matrix to a binary matrix
// by setting entry i,j to "1" if the rank-1 count is above the threshold and 0 
// otherwise. This binary matrix can be run through a "transitive closure" operation, or
// "connected component" operation to "complete" the clustering. 

// Inputs
// vecs -        Input FxN matrix of CNN vectors.
// gallery -     gallery vectors: FxG
// G -           number of gallery images. Not the same as gallerySize (50).
// N -           Number of images to be clustered (length of row of vecs).
// F -           Number of features per image (length of column of vecs) (4096).
// K -           How many nearest neighbors of each feature to consider? (a fraction of G)
// gallerySize - How large of a gallery to simulate? (50)

// Output
// bitMat -      (float **) that is NxN with counts.

float **Rank1Count(float **vecs,float **gallery,int G,int N,int F,int K,int gallerySize) {

    // Do some argument checking.
    if (gallerySize>G) {
    	gallerySize=G;
    	printf("Warning: resetting simulated gallery size to number of gallery images (G).\n");
    }
    
    // The "rank" of B with respect to A says what neighbor B is of A compared
    // to the gallery images. If B is closer to A than all gallery images, it is
    // neighbor 1 (not 0). If B is further from A than all gallery images, it is
    // neighbor G+1.
    // 
    // The table expProb (see below) gives the probability that, for a randomly
    // chosen subgallery of size gallerySize, the Bth neighbor would be rank 1 among
    // the sampled subgallery. We assume that there are no tied distances in this
    // computation. 
    // This is computed as:
    //   (1.0- (neighbor-1)/G)^gallerySize.
    // Thus, if B is neighbor #1, the probability that a random subgallery will 
    // make B rank1 is 1.0, since there is no chance of picking a gallery image
    // that will be closer to A than B is.
    // 
    // If B is neighbor G+1, then the prob. of rank1 is 0, since there is no
    // way that a selection of subgallery images (even sampling with replacement)
    // can leave B closer to A than (any of) the gallery images.
    
    // Relationship between K and G.
    // When examining the neighbors of A, we want to keep traversing the list (in
    // both directions) until we have either looked at every cluster image (condition 1) or
    // we have looked at enough *gallery* images so that the rank1 probability 
    // of any remaining images is neglible (condition 2). There is no problem
    // in continuing to examine cluster images after all gallery images have been seen,
    // at which point neighbor#=G+1. However, K would usually be set much smaller than
    // G+1, so that we would terminate before we get to this point.
    // 
    // Condition 1 is checked by keeping a count of how many cluster images have been given 
    // a rank1 probability. Since we start with A, which is one of the cluster images, 
    // clusterImgCnt is initialized to 1. Then, it is incremented after each calculation
    // of a rank1 probability. When clusterImgCnt reaches N, we should terminate the loop.
    // 
    // Condition 2 is determined by K. When the neighbor rank == K, the probability of 
    // rank1 becomes neglible. Typically K would be no more than 0.2*G.
    if (K>G) {
    	K=G;    // Cannot consider more neighbors than are in the gallery.
        printf("Warning: resetting neighbor count to gallery size.\n");
    }

    // Allocate the output matrix.
    float **bitMat=new float*[N];
    for (int i=0; i<N; i++) 
        bitMat[i]=new float[N];

    // Produce table of expected probabilities of rank-1, given kth nearest
    // neighbor index and gallery size.
    // Notice that the biggest rank something can have is G+1, if it is 
    // bigger than all G of the gallery images. Thus, ranks can go from 1 to
    // G+1. We fill in "rank 0", which can never be used, so that we can index
    // the array using rank instead of (rank-1).
    float *expProb=new float[G+2];                     // Need values 0-(G+1).
    double galSize=(double) gallerySize;
    expProb[0]=1.0;    // This is a dummy value and should never be used.
    for (int i=1; i<=G+1; i++) {
        double probGalleryGreater= 1.0-((double) i-1)/((double) G);
        expProb[i]=pow(probGalleryGreater,galSize);
    }

    // Copy and sort each row of incoming vectors. Produce a set of indices at same time.
    // For each feature, we need an array "index" which takes as an argument the position in 
    // the sorted list and returns the index of the value in the original list.
    // This can be done by simply attaching the original index to the original value
    // using the structure defined above.
    
    IndexedFloat **sortvecs=new IndexedFloat *[F];
    int **destind=new int *[F];

    for (int f=0; f<F; f++) {
        sortvecs[f]=new IndexedFloat [N+G];   // Each row contains clust and gal feats.
        // First copy cluster images.
        for (int j=0; j<N; j++) { 
            sortvecs[f][j].val=vecs[f][j];       // Images to be clustered.
            sortvecs[f][j].index=j;
            sortvecs[f][j].source=CLUSTER;
        }
        // Now copy gallery images.
        for (int j=0; j<G; j++) { 
            sortvecs[f][j+N].val=gallery[f][j];   // Gallery images.
            sortvecs[f][j+N].index=N+j;
            sortvecs[f][j+N].source=GALLERY;
        }
        
        qsort(sortvecs[f],N+G,sizeof(IndexedFloat),compare_IndexedFloats);
        // destind[f][foo] says where will element foo end up AFTER the sort.
        destind[f]=new int[N+G];
        for (int j=0; j<N+G; j++) 
            destind[f][sortvecs[f][j].index]=j;
    }

    float MAX_FLOAT=std::numeric_limits<float>::max();
    
    int leftInd,rightInd,origIndex;
    float leftDif,rightDif;
    float *accumRanks=new float[N];
    for (int i=0; i<N; i++) {              // For each clustering image...
    // To select the "A"'s, in the pairs (A,B), work with the UNSORTED ROWS (urow).
    // That way, the entire column of features comes from the same vector.
    // However, in comparing these values to other elements of the row, 
    // use the SORTED row (srow).
        for (int j=0; j<N; j++)              //   Initialize accumRanks vector.
            accumRanks[j]=0.0;
        for (int f=0; f<F; f++) {            //   for each feature...
            float *uRow=vecs[f];                   // unsorted row.
            IndexedFloat *sRow=sortvecs[f];        // sorted row.
            leftInd=destind[f][i]-1;   
            rightInd=destind[f][i]+1;  
            float pivotVal=uRow[i];                // value we're comparing distance to. 
      
            leftDif=(leftInd==-1)?MAX_FLOAT:pivotVal-sRow[leftInd].val;
            rightDif=(rightInd==N+G)?MAX_FLOAT:sRow[rightInd].val-pivotVal; //rightDif=(rightInd==N)?MAX_FLOAT:sRow[rightInd].val-pivotVal;

            int rank=1;
            int clusterImCnt=1;
            // Proceed through each row until 
            //   a) we have found K elements of cluster images closest to A (rank>K)
            //      or
            //   b) we have processed N cluster images. (clusterImCnt==N)
            while (rank<=K && clusterImCnt<N) {
                if (leftDif<rightDif) {
                    if (sRow[leftInd].source==CLUSTER) {              // source == CLUSTER
                        origIndex=sRow[leftInd].index;        // Where was feature in unsorted vec?
                        accumRanks[origIndex]+=expProb[rank];
                        clusterImCnt++;
                    }
                    else rank++;                                      // source == GALLERY
                    // Now adjust the leftInd "pointer" to the next el.
                    if (--leftInd==-1) 
                        leftDif=MAX_FLOAT;
                    else leftDif=pivotVal-sRow[leftInd].val;
                }
                else {          //        (rightDif>=leftDif)            
                    if (sRow[rightInd].source==CLUSTER) {        // source == CLUSTER          
                        origIndex=sRow[rightInd].index;
                        accumRanks[origIndex]+=expProb[rank];
                        clusterImCnt++;
                    }
                    else rank++;                                 // source == GALLERY
                    // Now adjust the rightInd "pointer" to next el.
                    if (++rightInd==N+G)
                        rightDif=MAX_FLOAT;
                    else rightDif=sRow[rightInd].val-pivotVal;
                }
            }
        }
        // We just finished processing all of the features for a given pivot.
        // Now, we want to take the accumRanks vector, which is a floating point
        // array of size N, and store it as one of the rows in a more compact
        // final matrix.
        for (int j=0; j<N; j++) {
            if (accumRanks[j]<0.0)
                accumRanks[j]=0.0;
            bitMat[i][j]=accumRanks[j];
        }
    }
  
    
    // Deallocate where necessary.
    printf("starting deleteions\n");
    delete expProb;
    for (int i=0; i<F; i++) delete [] sortvecs[i];
    delete accumRanks;
    printf("accumrnaks\n");


    return bitMat;
}

int compare_IndexedFloats (const void *a, const void *b) {
    // We cheat a bit here to make things faster. 
    // We pretend we are dereferencing a float ptr even
    // though we are really dereferencing a ptr to an IndexedFloat.
    // We can get away with this since the first element of an IndexedFloat
    // is the value we are looking for.
    // This saves us an extra dereference.
    float dif=(*(float*)a - *(float*)b);
    if (dif>0)
        return 1;
    else if (dif<0)
        return -1;
    else return 0;
}