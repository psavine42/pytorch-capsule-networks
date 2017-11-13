
#Capsulenet in Pytorch

Pytorch implementation of Capsule Network. https://arxiv.org/pdf/1710.09829.pdf 

#Run 
        git clone 
        cd
        python capsnet.py --dataset mnist --epochs 1

#Run on Floydhub



#Included 
Dataset loading for mnist, svhn, mnist.fashion

testing on MNIST as in paper. 
        first convolutional layer channels -> 64, 
        primary capsule layer to 16 6D-capsules 
        final capsule layer 8D at the end and 
        achieved 4.3% on the test

testing on SVHN as in paper.
        first convolutional layer channels -> 64, 
        primary capsule layer to 16 6D-capsules 
        final capsule layer 8D at the end and 
        achieved 4.3% on the test

#Todo
clean up and documentation.

other papers: https://openreview.net/pdf?id=HJWLfGWRb