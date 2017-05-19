# Feed Forward Networks: Hierarchical Language Model

In Professor Hinton’s lecture(Lecture 4d of course Neural Network for Machine
Learning), he mentioned a very interesting topic: learning to predict the next
word by predicting a path through a tree. This post aims to go through two
related papers and distill some inspiring ideas.

## Background

In previous two posts [Forward Propagation for Feed Forward
Networks](https://medium.com/@SauceCat/forward-propagation-for-feed-forward-networks-ac8fcb6bdd60)
and [Backward Propagation for Feed Forward
Networks](https://medium.com/@SauceCat/backward-propagation-for-feed-forward-networks-afdf9d038d21),
we have gone through both forward and backward propagation process of the simple
feed forward networks. Actually, the toy example is based on Bengio’s NPLM
([Neural Probabilistic Language
Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)) in 2003. The
most notable difference is that Bengio’s net contains more than 15,000 output
units while our simple net only has 250.

#### Bengio’s net in 2003
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*rPB6sk3hn4qPmlgmkVFzpQ.png'></p>

Mathematical representation for this predictor is as follow:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*ihEPGnIl_F9_0sPG8bNAIA.png'></p>

where x is the concatenation of the input word feature vectors, W represents the
direct connections between the input layer and the output layer and is
optionally zero (no direct connections).

However, as the vocabulary size growing large, NPLM becomes very slow to train
and test. Computing the probability of the next word requires normalizing over
all words in the vocabulary. What’s worse, calculating exact gradient needs to
do this computation repeatedly to update the model parameters iteratively. Thus,
it is very time consuming. One potential improving method, as mentioned in the
lecture, is to learn in the serial architecture using the following network
structure.

#### Learning in the serial architecture
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*HDxsbWyFAaWnUEsra0ikpQ.png'></p>

For each context, one can first compute the logit score for each candidate next word and then normalize all the logits using softmax function to get the probabilities.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*wEPnRocDo5pGX1tUQ6FYTQ.png'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*6iO2yYXrt8xV-WirLNuwEg.png'></p>

If there are only limit number of candidate words, one can save even more time. For example, the candidates can be limited in the set of possible words suggested by the trigram model.

## Hierarchical Probabilistic Neural Network

This part is based on Morin and Bengio’s paper *[Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf)*.

This paper proposes a much faster variant of the original HPLM. The basic idea
is to construct a hierarchical description of a word by arranging all the words
in a binary tree with words as the leaves (each tree leaf is associated with one
word). Then the task becomes to learn to take probabilistic decision at each
tree node so as to get down to the target word (each tree node represents a
cluster of words).

### Hierarchical decomposition

In formal representation, instead of computing **P(Y|X)** directly, we can
define a clustering partition for **Y** with a deterministic function **c(.)**
mapping **Y** to cluster **C**, then **P(Y|X)** can be decomposed as:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*xWj6o_XW7swF_DTYgZTG0g.png'></p>

The proposed binary tree is actually a binary hierarchical clustering of words,
which can be interpreted as a series of binary stochastic decisions associated
with nodes. Each word can be represented by a bit vector **(b1(v), b2(v),…,bm(v))**,
where **m** is the number of decisions needed to take to reach the target word. For
example, according to the tree structure below, the word “cat” can be
represented as (1, 0). (1 means to take the left node) And we can see that words
“dog” and “cat” belong to the same second-level word cluster, while “tree” and
“flower” belong to the other.

#### Simple binary hierarchical clustering of words
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*U_6aw-XOezKgamj231mK4Q.png'></p>

Then, conditional probability for a next word **v** can be represented as
sequential prediction down a tree path.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*QA0f8DcJyV7hzMKNWb28mw.png'></p>

In this way, for each training sample, there will be target and gradients only
for the path associated with the target word. This is the major source of time
saving during training.

### Sharing parameters across the hierarchy

The other trick to reduce computation is to share parameters across the
hierarchy tree. First, it is reasonable to share the word embedding across all
nodes. Second, it also makes sense to associate each node with a feature vector
similar to that for word embedding because each tree node is actually
representing a cluster of similar words. Then we would have feature vectors for
both words and tree nodes.

With all these assumptions, at each tree node, we can consider the model to
predict the next tree node based on two kinds of input: the context and the
current tree node (more precisely, is the concatenated feature vectors of the
context words and the feature vector of the current tree node). This can be
represented by a model similar to NPLM but with two kinds of input.

#### predictor at each tree node
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*8p1z3M2vyf5sFV2SQtp1ug.png'></p>

The predict function can be written as:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*2xIPOWw60IX2vnUfGQW1zQ.png'></p>

We can see that this formula is quite similar to the following one, which is
from the original NPLM. The major difference is that the feature vector of
potential next word **v** is replaced by the feature vector of the current node.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*wEPnRocDo5pGX1tUQ6FYTQ.png'></p>

### Using WordNet to build the hierarchical decomposition

A very important component of this model is to construct the hierarchical binary
word tree. This paper proposes to utilize the prior knowledge from the
[WordNet](https://wordnet.princeton.edu/) resource.

> **WordNet** is a [lexical
> database](https://en.wikipedia.org/wiki/Lexical_database) for the English
language. It groups English words into sets of
[synonyms](https://en.wikipedia.org/wiki/Synonyms) called
[synsets](https://en.wikipedia.org/wiki/Synsets), provides short definitions and
usage examples, and records a number of relations among these synonym sets or
their members.

> — Wikipedia

The IS-A taxonomy in WordNet is already a hierarchical tree of words classified
by semantic type. However, some tree nodes have multiple parents and some words
are associated with multiple leaves (because leaves are actually senses, it is
common that one word is related to multiple senses). Some simple modifications
are necessary to make the tree useful: manually select one parent for each of
the few nodes associated with several parents and assign each of the few words
associated with multiple leaves with only one leaf.

#### noun taxonomy graph from WordNet
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/0*znQMF_VdRS6KjRH2.'></p>

*(The graph above is from kleem’s
*[GitHubGist](https://gist.github.com/kleem/6ab92f48ef961da271ab)*, really
informative visualization!)*

#### fragment of WordNet taxonomy graph
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*u90KLC0Wk7YcHsAeirYP3Q.png'></p>

*(This example is based on the post
*[here](http://www.clips.ua.ac.be/pages/pattern-search)*. In this example, words
“tiger” and “lion” share the same parent because they are both “big cat”. Words
“dog”, “wolf” and “fox” are grouped together since they are all “canine”.)*
<p align='center'><img src='https://cdn-images-1.medium.com/max/600/1*O_Jy1LjQmM4TQjZgy9sbPw.png'></p>

The last step is to turn the modified tree into a binary one. The proposed
method is to perform a data-driven binary hierarchical clustering of the
children associated with each node using simple K-means algorithm. To compare
nodes, each tree node is associated with the subset of words that it covers.
Since each word in the vocabulary set can be associated with a TF-IDF vector
calculated from the training corpus, each tree node can be reasonably
represented by the dimension-wise median of the TF-IDF vectors of the subset of
words it covers.

### Conclusion

The implementation and the experiments show that the proposed model
significantly speeds up the learning process when the number of output classes
grows huge. But it doesn’t generalize as well as the original NPLM. However,
given the very large speed-up, it is certainly worth further investigation. One
promising point is to fully utilize the word sense ambiguity information
provided by WordNet’s taxonomy by allowing one word to be associated with
multiple senses (tree leaves).

## A Scalable Hierarchical Distributed Language Model

This part is based on Mnih and Hinton’s paper [A Scalable Hierarchical
Distributed Language
Model](https://pdfs.semanticscholar.org/1005/645c05585c2042e3410daeed638b55e2474d.pdf).

This paper claims the main limitation of the hierarchical model mentioned above
is the precedure of tree construction. It proposes an automated method for
building tree directly from the traing data without any kind of prior knowledge.

### The log-bilinear model

The simple log-bilinear language model (LBL) is used as the building block for
the proposed model. To predict the next word w_n based on the context w_1 to
w_n-1, the model first calculates the predicted feature vector (with same
dimension as the embedding feature vector) for the next word by just simply
linearly combining the feature vectors of the context words:

Then the similarities between all candidate words and the predicted feature
vector can be computed using the inner product. The similarities are further
exponentiated and normalized to obtain the distribution over the next word.

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Is5YDuuqIuaUsrzCl18R1g.png'></p>

> The LBL model can be interpreted as a special kind of a feed-forward neural
> network with one linear hidden layer and a softmax output layer. The inputs to
the network are the feature vectors for the context words, while the matrix of
weights from the hidden layer to the output layer is simply the feature vector
matrix R. The vector of activities of the hidden units corresponds to the the
predicted feature vector for the next word. Unlike the NPLM, the LBL model needs
to compute the hidden activities only once per prediction and has no
nonlinearities in its hidden layer.

> — A Scalable Hierarchical Distributed Language Model

The NPLM needs to compute the hidden activities once for each decision since the
feature vector of the tree node, which keeps changing down the path, is included
in the model input. However, in the LBL model, the hidden layer is replaced with
the predicted feature vector. Thus, the LBL model can save a lot of time by
calculating the hidden activities once per prediction (one prediction contains a
sequence of decisions).

#### log-bilinear model
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*8_5qDCaQGk-AZhiDHEy0Pw.png'></p>

### The hierarchical log-bilinear model

The hierarchical model here is based on Morin and Bengio’s idea but uses LBL to
compute provabilities at each node and allows each word to be associated with
multiple leaves.

The probability of the next word being w is the joint probability of making
sequence of binary decisions specified by the word’s code. To make things
simple, let’s first assume that each word is associated with exactly one leaf.
Then each word is actually corresponding to only one path down the tree, which
can be described by a binary string. For example, string “10” means to take left
child node at the first node and then go to right child node at the second node.
The probability of the next word can be written as a product of probabilities of
the binary decisions:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*8HtLx3PbnZ8ZNokEs5HCJw.png'></p>

where d_i is the i_th digit in the code for word w, and q_i is the feature
vector for the i_th node in the path corresponding to that code. In details, the
probability of each decision is given by:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*_eOU-HPKDuAAl9JrnFXZUQ.png'></p>

When extending to words with multiple codes (each word can relate to more than
one leaf), the equation becomes:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*_zDDkdV6MIPvEezkohFAtQ.png'></p>

where D(w) is a set of codes associated with word w.

### Hierarchical clustering of words

In terms of tree construction, it is proposed to use a pure learning approach
instead of using expert knowledge like the IS-A taxonomy DAG from WordNet. In
other words, it prefers to implement hierarchical clustering algoithm directly
on the training data. With this idea in mind, we need to consider how to
represent each word in the vocabulary set.

Before really going into the solution, we can first think about what we want
from the clustering of words. Well, we would like to make it easilier to predict
the next word based on the context. Thus, it is quite natural to consider
describing each word in terms of the contexts that can precede it. Certainly we
can use the distribution of contexts that can precede the target word. However,
as the context size grows, the number of contexts that might potentially procede
the target word would grow exponentially. Thus, it is not a good idea. We want
condense, low-dimensional real-valued representation.

The predicted vector could be a reasonable choice. It is generated based on the
context and has much smaller dimension compared with the vocabulary size. Thus
it is condense and can somehow represent the information provided by the
context. However, to generate the predicted feature vector, we need to train the
HLBL with a hierarchical tree. The paper proposes a really clever bootstrapping
procedure: **first generate a random binary tree of words, then train a HLBL
based on the random tree, and finally use the learned predicted feature vectors
to reconstruct the tree**.

In this way, each context would finally be associated with a learned predicted
feature vector. Since there are many possible contexts can precede to a specific
next word, each word can be associated with a number of predicted vectors
generated by all possible contexts. It is proposed to summarize all these
predicted feature vectors for a specific word by computing the expected
predicted feature vectors w.r.t. the distribution of all possible contexts. Then
the word is finally represented by this expected vector.

#### Generate the expected predicted feature vector
<p align='center'><img src='https://cdn-images-1.medium.com/max/1000/1*-ry1UxybNNHfBDi8C023uQ.png'></p>

With the learned representation for each word, at each step, simple
two-component Gaussian mixture clustering algorithm is implemented by runing the
EM algorithm for 10 steps. Then the words in current set are partitioned into
two subsets based on the responsibilities of the two mixture components for
them. (Split rules will be discussed later) The fitting and partition continue
until the current set contains only two words. Note that randomization exists
inside the Gaussian mixture model, thus different runs might produce quite
different clusters. To suppress the influence of the randomization, one can
build several trees based on different runs and collect these trees together to
form a big tree. Then the model would have flexibility to choose between
multiple possible clusterings.

#### Big tree from 4 different runs
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Wd-_bF58ZGjtaJhiOW9quw.png'></p>

Another thing to consider is the split rules (how to split the two subsets). The
goal is to generate a tree that is both well-balanced and well-supported by the
data. To explore the trade-off between these two requirements, three different
split rules are proposed for further testing:

**Rule 1.** **Build a balanced tree at any cost:** sort the responsibilities and
split the words into two disjoint subsets of equal size based on the sorted
order.

**Rule 2. Build a tree that is totally based on the data:** assign a word to the
component with higher responsibility for it.

**Rule 3. Extension of the second rule to make it balanced:** assign a word to
both component whenever both responsibilities are within **Epsilon** of 0.5. For
example, if **Epsilon** is set to 0.1, then if the responsibilities are between
0.4 and 0.6, then the word is assigned to both component. In this way, some
words would be associated with multiple paths down the tree, informing that they
are difficult to cluster.

The experiments show that the third rule with a carefully selected **Epsilon
**(0.4 in the paper) as well as with multiple runs (4 in the paper) outperforms
other rules.

### Conclusion

The paper concludes that, the key to build a well-perform hierarchical model is
using a carefully constructed hierarchy over words. Creating hierarchies in
which every word can be associated with multiple paths is essential to getting
the models to perform better (which is also proposed in the conclusion part of
Morin and Bengio’s paper).

In fact, when I first came across the bootstrapping procedure proposed by the
paper, I started to think about why not repeat the procedure for several times
so the tree structure could also be updated through the learning process. And
finally I found out that this idea is proposed in the last paragraph of the
paper. :)

> Finally, since our tree building algorithm is based on the feature vectors
> learned by the model, it is possible to periodically interrupt training of such
a model to rebuild the word tree based on the feature vectors provided by the
model being trained. This modified training procedure might produce better
models by allowing the word hierarchy to adapt to the probabilistic component of
the model and vice versa.
