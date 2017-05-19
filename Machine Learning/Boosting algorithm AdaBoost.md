# Boosting algorithm: AdaBoost

As a data scientist in consumer industry, what I usually feel is, boosting
algorithms are quite enough for most of the predictive learning tasks, at least
by now. They are powerful, flexible and can be interpreted nicely with some
tricks. Thus, I think it is necessary to read through some materials and write
something about the boosting algorithms.

Most of the content in this acticle is based on the paper: [Tree Boosting With
XGBoost: Why Does XGBoost Win “Every” Machine Learning
Competition?](https://brage.bibsys.no/xmlui/handle/11250/2433761). It is a
really informative paper. Almost everything regarding boosting algorithms is
explained very clearly in the paper. So the paper contains 110 pages :(

For me, I will basically focus on the three most popular boosting algorithms:
AdaBoost, GBM and XGBoost. I have divided the content into two parts. The first
article (this one) will focus on AdaBoost algorithm, and the second one will
turn to the comparison between GBM and XGBoost.

## AdaBoost

AdaBoost, short for “Adaptive Boosting”, is the first practical boosting
algorithm proposed by Freund and Schapire in 1996. It focuses on classification
problems and aims to convert a set of weak classifiers into a strong one. The
final equation for classification can be represented as

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*B2987FKIw3QL2ClYR_OeuQ.jpeg'></p>

where **f_m** stands for the **m_th** weak classifier and **theta_m** is the corresponding
weight. It is exactly the weighted combination of **M** weak classifiers. The whole
procedure of the AdaBoost algorithm can be summarized as follow.

### AdaBoost algorithm

Given a data set containing **n** points, where

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*2fp-O3KfXqrdYEGU_RjY0w.jpeg'></p>

*Here -1 denotes the negative class while 1 represents the positive one.*

Initialize the weight for each data point as:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*IMHTVrXPKc2mVqDDK40k9w.jpeg'></p>

### For iteration m=1,…,M:

(1) Fit weak classifiers to the data set and select the one with the lowest
weighted classification error:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*C8-yNia8Oh44X-t0UxUCUA.jpeg'></p>

(2) Calculate the weight for the **m_th** weak classifier:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*jFpUGuxpGZuzpG6FlDAASw.jpeg'></p>

*For any classifier with accuracy higher than 50%, the weight is positive. The
more accurate the classifier, the larger the weight. While for the classifer
with less than 50% accuracy, the weight is negative. It means that we combine
its prediction by flipping the sign. For example, we can turn a classifier with
40% accuracy into 60% accuracy by flipping the sign of the prediction. Thus even
the classifier performs worse than random guessing, it still contributes to the
final prediction. We only don’t want any classifier with exact 50% accuracy,
which doesn’t add any information and thus contributes nothing to the final
prediction.*

(3) Update the weight for each data point as:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*mqLcX8yookiPVZoAe6iwqA.jpeg'></p>

where **Z_m** is a normalization factor that ensures the sum of all instance weights
is equal to 1.

*If a misclassified case is from a positive weighted classifier, the “exp” term
in the numerator would be always larger than 1 (y*f is always -1, theta_m is
positive). Thus misclassified cases would be updated with larger weights after
an iteration. The same logic applies to the negative weighted classifiers. The
only difference is that the original correct classifications would become
misclassifications after flipping the sign.*

After **M** iteration we can get the final prediction by summing up the weighted
prediction of each classifier.

## AdaBoost as a Forward Stagewise Additive Model

*This part is based on paper: *[Additive logistic regression: a statistical view
of
boosting](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)*.
For more detailed information, please refer to the original paper.*

In 2000, Friedman et al. developed a statistical view of the AdaBoost algorithm.
They interpreted AdaBoost as stagewise estimation procedures for fitting an
additive logistic regression model. They showed that AdaBoost was actually
minimizing the exponential loss function

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Yma0SQlBEFiOMvpkyssZsw.jpeg'></p>

It is minimized at

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*MqKgtWPJdX0VLuU5GO-TNw.jpeg'></p>

Since for AdaBoost, **y** can only be -1 or 1, the loss function can be rewritten as

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*wJ4oTBoHtct5ic0pHKMnfA.jpeg'></p>

Continue to solve for **F(x)**, we get

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*YrJRjDF8W5z06YltBJWHPw.jpeg'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*2FdLn5qyQQBwcUSdJlPgww.jpeg'></p>

We can further derive the normal logistic model from the optimal solution of **F(x)**:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*tW70xptkSSh7f7kb23-Faw.jpeg'></p>

It is almost identical to the logistic regression model despite of a factor 2.

Suppose we have a current estimate of **F(x)** and try to seek an improved estimate
**F(x)+cf(x)**. For fixed **c** and **x**, we can expand **L(y, F(x)+cf(x))** to second order
about **f(x)=0**,

Thus,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*08YHFqFZW-JuTqOpbDLBRA.jpeg'></p>

where **E_w(.|x)** indicates a weighted conditional expectation and the weight for
each data point is calculated as

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*aNN9b_CNa2GfQfrKfrpvog.jpeg'></p>

If **c > 0**, minimizing the weighted conditional expectation is equal to maximizing

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*XnbrviYa0W3yvORj2N6z5w.jpeg'></p>

Since **y** can only be 1 or -1, the weighted expectation can be rewritten as

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*wuDY2LwDvM_ewk_I7ldDsw.jpeg'></p>

The optimal solution comes as

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*IU_ynTQMpTz_ZCIQvg_AnQ.jpeg'></p>

After determining **f(x)**, the weight **c** can be calculated by directly minimizing **L(y, F(x)+cf(x))**:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*a3xUJLSTv8FWwGEgE8xM8A.jpeg'></p>

Solving for **c**, we get

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*oU1Qwkp2ACByWQGVg38IWA.jpeg'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*oS3XycIubLutkV9d4uMkMw.jpeg'></p>

Let epsilon equals to the weighted sum of misclassified cases, then

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*mXzeLgnEi_D7t2VTVp5TDQ.jpeg'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*ol_jhn68ALTYAGTsTOrkAg.jpeg'></p>

Note that **c** can be negative if the weak learner does worse than random guess
(50%), in which case it automatically reverses the polarity.

In terms of instance weights, after the improved addition, the weight for a
single instance becomes,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*PKA_SiqiCtsxLlmzls8uDw.jpeg'></p>

Thus the instance weight is updated as

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*P8lqZFNAlYPF8nQmcuIqTA.jpeg'></p>

Compared with those used in AdaBoost algorithm,

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*oMtsmZhRaKdVWlZSilkDJg.jpeg'></p>

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*jFpUGuxpGZuzpG6FlDAASw.jpeg'></p>

we can see they are in identical form. Therefore, it is reasonable to interpret
AdaBoost as a forward stagewise additive model with exponential loss function,
which iteratively fits a weak classifier to improve the current estimate at each
iteration **m**:

<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*iebd6Q_Lda4yEtPTnj6u7Q.jpeg'></p>

