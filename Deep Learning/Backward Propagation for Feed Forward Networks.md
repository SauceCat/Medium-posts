# Backward Propagation for Feed Forward Networks

This is the continuation of the previous post [Forward Propagation for Feed
Forward
Networks](https://medium.com/@SauceCat/forward-propagation-for-feed-forward-networks-ac8fcb6bdd60).
After understanding the forward propagation process, we can start to do backward
propagation.

## The error function (the cost function)

To train the networks, a specific error function is used to measure the model
performance. The goal is to minimize the error(cost) by updating the
corresponding model parameters. To know which direction and how much to update
the parameters, their derivatives w.r.t the error function must be calculated.
And this is what backward propagation is used for.

The choice of a error function is always closely related to the specific use
case. Well, for this case, as mentioned in the previous post, the output layer
is activated by the softmax function,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*hWGSjuZlSCCm-pIVd9ncHQ.png'></p>

For the softmax function, cross entropy error is a well-known choice. The reason
might be that, as I guess, the cross entropy error is specially designed for
probability outputs, while other error function, like the square error, is for
general cases. Besides, unlike the misclassification error, cross entropy error
measures the model performance more comprehensively. Because, for example, for
the binary case, assigns 0.6 or 0.7 to the wrong class yields the same model
performance when the performance is measured by misclassification error, while
the cross entropy error would prefer the first model rather than the second one.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*l6_9tqMqLWbaEXaNY9L9cg.png'></p>

## Backward Propagation

### Error to output net layer

It is worth to point out that, because of the specialty of the softmax function,
each output net unit is connected to all output activate units. Thus, during
back propagation, each output net unit receives errors from all output activate
units.

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*9zrgf_qTm0bzFLphXIJemA.png'></p>

For a single output net unit *k* with a single training sample:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*sf1n6GgoZdQ9_q4jpJOWlg.png'></p>

Further, the equation can be decomposed into two parts. For the output net unit,
there are two kinds of derivatives, depending on whether the error is propagated
from the output activate unit *k* or others.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*g2JMSr--83_N8AS7XXTzVQ.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*513qS0KchosLeQuCSGOo-g.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*LE-eFvcR6jrgRjuNgkoW4w.png'></p>

Finally it comes out that the derivative is in surprisingly simple format. Using
matrix representation, the equation becomes even more elegant for a batch of
training samples.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*sQDBv9irY9nZWwHx3Qc2tg.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*QZKPjqc3UTaCIrDtYfVqLQ.png'></p>

### Error to hidden_to_output weights

Based on the previous post, for a single output net unit k with one training
sample,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*zbFdAAzXMsvSoLUUJg07hA.png'></p>

Then, for a single weight with a single training sample,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*csWd73oGJhIxHa1lh55m6A.png'></p>

When it goes to a batch of training samples,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*dwLLHN_NbUGg0BY4OhoCYw.png'></p>

For all weights connect the hidden layer with the output layer, using matrix
representation,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*AqFaMNqv-HZsIHtkEeCfbQ.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*UT4eDTLpML4WeqAFXcnifw.png'></p>

For the output bias term, we can treat it as a weight always connected with
hidden state ‘1’. Then, we can also generate the derivative for those biases.
For example, for the bias term of the output net unit *k*,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*7a76zTt-6gT8it4ABdpbMA.png'></p>

We can generate derivatives for all output bias terms by suming across the rows
of the output net unit error matrix.

### Error to hidden net layer

For the hidden net layer, each hidden activate unit receives errors from all
output net units and then passes the aggregated error back to its corresponding
hidden net unit.

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*lDEeyZxcrA5lI1MLo3DAOQ.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*O4qWfoOjT5frx352y2LwEw.png'></p>

For the sigmoid function,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*qaogvq7SCblGh5H6IhLSfA.png'></p>

The derivative comes to be very simple and elegant,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*m4pHt-7A_8sHyinXIb9MWA.png'></p>

The back propagated error for the hidden net unit becomes,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*tI_k_K4u7mVtaaZ8QHZxxQ.png'></p>

For all hidden net units with a batch of training samples, the derivatives
become,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*dLSomzy__dZZeFTubXoI-g.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*g-R7LAPJ_vImLiOThrkWfA.png'></p>

### Error to input_to_hidden weights

According to the previous post,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*A5t5rLzvU4KqWPj180bC1A.png'></p>

Then, for a single weight *w_ji*,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*f5OLJLmow5XSxeVRX4LL2w.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*TEOmGJftqeUMc_Hsh_WYXw.png'></p>

For all weights with a batch of training samples,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*zDt1O_pV8hm-_SqytXK6Lw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*7R_p72a_I5-ZnR0CXd5BLA.png'></p>

Also, just as mentioned before, in terms of the hidden unit bias, we can view it
as a weight always connected with input unit with value ‘1’.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*DK43oidanhbt_Z9jzIQgfw.png'></p>

### Error to input layer

Because we don’t have any activate function for the input layer, there is no
input activate layer. Each input unit just receives errors directly from all
hidden net units.

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*V3cZ8_kJ8nPPg_ieMMkkew.png'></p>

According to the previous post,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*A5t5rLzvU4KqWPj180bC1A.png'></p>

Thus, for a single input unit *i* with a single training sample,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*zjngO_f_1xvCj8wEF0BUfg.png'></p>

Then for all input units with a batch of training samples (there is no activate
function for the input layer),

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*d0N1qEpJT9cwYWaQMNp5zw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*SH-EREQEOH42gHHW3rSYEQ.png'></p>

## Summary

To summarize, let’s put forward propagation and backward propagation together
here.

### Forward Propagation

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*ZQ31NUy8Xwu0JDFz2XgCtw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*az86P-EybXm77Y0I31N8-Q.png'></p>

### Error function

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*l6_9tqMqLWbaEXaNY9L9cg.png'></p>

### Backward Propagation

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*sQDBv9irY9nZWwHx3Qc2tg.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*AqFaMNqv-HZsIHtkEeCfbQ.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*7a76zTt-6gT8it4ABdpbMA.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*dLSomzy__dZZeFTubXoI-g.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*zDt1O_pV8hm-_SqytXK6Lw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*DK43oidanhbt_Z9jzIQgfw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*d0N1qEpJT9cwYWaQMNp5zw.png'></p>

### Learning Update

With the mini-batch learning algorithm, all parameters are updated once after a
full scan of a batch of training samples. Thus, during the update, all gradients
generated above must be divided by the batch size. Besides, in this case, the
momentum method is used.

**Word embedding layer (input layer)**

This layer is a little bit complicated, because not all word embedding vectors
need to be updated for each of the training sample. For a batch of training
samples, we must first carefully make up a update matrix with shape the same as
the word embedding matrix. Initially, all elements of the update matrix are
zeros. For each training sample, we assign or aggregate the gradient for the
related word embedding weights in the corresponding row of the update matrix.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*1YcAk4uR5Pf8AFSDGGbjNA.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*wxoA29PPJG-JUdEN_oiOeA.png'></p>

**embed to hidden layer weights**

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*jVZj0ISBgTxf-TboL-4sUw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Y0UPVVrcQCPRLs4rDCC91w.png'></p>

**hidden to output weights**

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Lw1D5L33USrTMrAH1nTF2w.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*tJbhizJ8e-5_duahIUXl9A.png'></p>

**hidden bias**

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Ky7qZMBVmyo5Nu16UEsbdw.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*awR-3-r6RvLs4cw0uO9mHw.png'></p>

**output bias**

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*HLxADKnZocfm2GJxCX-Heg.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*b8Kb2o0pIFDGGkgB5u3sxA.png'></p>

