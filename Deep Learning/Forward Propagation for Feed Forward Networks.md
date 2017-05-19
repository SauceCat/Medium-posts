# Forward Propagation for Feed Forward Networks

During the study of Professor Geoffrey Hinton’s online course (Neural Networks
for Machine Learning), I found that the best way to understand a type of
networks is to spend some time going thoroughly the forward and backward
propagation.

I want to start with the simplest but the most fundamental feed forward
networks. The specific network structure used in this post came from the
assignment of the course. Actually it is a toy example for the ‘learning word
representations’ task.

## The network structure

The goal is to predict the fourth word based on the previous three. The three
word embedding vectors, each with 50 elements, are concatenated according to
their appearing order and formed a 150-dimensional input vector. The hidden
layers contains 200 units and the output layer contains 250 units. All units
from different layers (input layer, hidden layer and output layer) are fully
connected to each other. (hidden net layer and hidden activate layer belong to
the same layer here)

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*DE8MiowzYi-MBv0508r8sA.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*ISXe4dwXiBFR3HQxl9XkGg.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*DVpxOW4MeC2vKGbnCps8kQ.png'></p>

## Forward Propagation

### Input layer to Hidden layer

Usually, for a simple feed forward network, only the weights need to be learned.
In this case, the word embedding vectors are also included in the learning
process. That’s why it is named ‘learning word representations’. For the toy
example, only 250 words are included in the dictionary.

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*mzXEjs8UIkEs0JJKI9zOoA.png'></p>

In practice, the input is actually in format **‘[word_index_1, word_index_2,
word_index_3]’**. These three indexes represents the row indexes of the
representation matrix.

When the mini-batch gradient descent algorithm is implemented, let N be the
batch size, the input vector is expanded into a ‘N-column’ input matrix.

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*OY_gp0T498PEl0ZZX9R7Uw.png'></p>

The hidden layer first aggregates all inputs from the input layer and then
activates the net values with sigmoid function.

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*Cgs1V0Ma8JPkyYb3KOdoug.png'></p>

For a single hidden unit *j* with a single training sample *n*:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*GmEBVXodE6KJpuklc6qx5Q.png'></p>

For all hidden units with a batch of training samples:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*ZQ31NUy8Xwu0JDFz2XgCtw.png'></p>

To understand the whole things clearly, I have visualized the process of the
matrix calculation (I used to struggle a lot here).

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*WB9esDWnuw_B1gN0prUHAA.png'></p>

First, the weight matrix is transposed. Then, the bias vector is expanded into N
columns. Finally, the calculation is visualized as:

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*5cGREfhe1RDYbL2FjPXwqA.png'></p>

Each column of the hidden matrix, which represents a hidden vector for a single
training sample, can be decomposed as the linear combination of the columns of
the weight matrix (the sum of contribution from all input units).

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*OGfbuYTWLXTNOolMp9fQPw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*TVpCk5B7_Q2ZLuRA21xsSg.png'></p>

### Hidden layer to Output layer

For the output layer, while other things remain almost the same, each unit of
the activate layer is connected with all units of the net output units because
of the softmax activate function.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*xEVe9wkF2fwKNAL7kBBIyA.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*IsmKhAXu9zBwl07pz_DugA.png'></p>

For a single output unit *k* with a single training sample *n*:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*D0m0BZRfVuwsVIk_yukxoQ.png'></p>

For all output units with a batch of training samples:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*az86P-EybXm77Y0I31N8-Q.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*auKHGy2fpsBZTd_4W7f_Ug.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*sUMPkfCT9sCoYViB0Rqxew.png'></p>

Each column of the output matrix, which represents a output vector for a single
training sample, can be viewed as the linear combination of the columns of the
weight matrix (the sum of contribution from all hidden units).

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*M_gOiTkvPSVl5Lx3vVWmVQ.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Sh1RNyaLuwJboVXfDUBIpA.png'></p>

## Summary

As we see, the forward propagation for the simple feed forward networks is quite
straightforward.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*ZQ31NUy8Xwu0JDFz2XgCtw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*az86P-EybXm77Y0I31N8-Q.png'></p>

Only one thing need to notice is the softmax activate function. It will become
quite tricky in the back propagation part, as you will see in the next post
about the back propagation of the feed forward networks.