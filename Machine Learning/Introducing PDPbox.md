# Introducing PDPbox

PDPbox is a partial dependence plot toolbox written in Python. The goal is to
visualize the impact of certain features towards model prediction for any
supervised learning algorithm. (now support all scikit-learn algorithms)

It is inspired by [ICEbox](https://github.com/kapelner/ICEbox) as well as
[PyCEbox](https://github.com/AustinRochford/PyCEbox). ICEbox is a R package for
Individual Conditional Expectation (ICE) plots, a tool for visualizing the model
estimated by any supervised learning algorithm. It is developed by one of the
authors of the paper: *[Peeking Inside the Black Box: Visualizing Statistical
Learning with Plots of Individual Conditional
Expectation](https://arxiv.org/abs/1309.6392)*, where individual conditional
expectation plots were introduced. While PyCEbox is a Python implementation of
individual conditional expecation plots.

PDPbox aims to wrap up and enrich some useful functions mentioned in the paper
in Python. To install PDPbox, please refer to
[PDPbox](https://github.com/SauceCat/PDPbox).

## The common problem

When using black box machine learning algorithms like random forest and
boosting, it is hard to understand the relations between predictors and model
outcome. For example, in terms of random forest, all we get is the feature
importance. Although we can know which feature is significantly influencing the
outcome based on the importance calculation, it really sucks that we don’t know
in which direction it is influencing. And in most of the real cases, the effect
is non-monotonic. We need some powerful tools to help understanding the complex
relations between predictors and model prediction.

### Friedman’s PDP

Friedman’s partial dependence plot aims to visualize the marginal effect of a
given predictor towards the model outcome by plotting out the average model
outcome in terms of different values of the predictor.

To formally define PDP, let **S** be the chosen predictor, and **C** be its
complete set (containing all other predictors), partial dependence function of
**f** on **x_S** is given by
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*xb7MZeuvSoBXNpTURKdHog.png'></p>

In practice, it is impossible to integrate over all possible values of **C**,
thus the formula above is usually estimated as
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*6tE2b9LC2KEvJV1HBgnnVQ.png'></p>

where **N** is the total number of observations in the training set. By
generating estimations on different values of **S**, we can get a line showing
how model prediction will change through different values of **S**.

For example, let’s assume a data set that only contains three data points and
three features (A, B, C) as shown below.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*xfp3l53TnMwzM2A9FubjXA.png'></p>

If we want to see how **feature A** is influencing the prediction **Y**, what
PDP does is to generate a new data set as follow and do prediction as usual.
(here we assume that feature A has three unique values: A1, A2, A3)
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*Fma2amdY2zq37Ck0EGe60A.png'></p>

In this way, PDP would generate **nrows * num_grid_points** number of
predictions and averaged them for each unique value of Feature A.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*bwLj7X84h_VOG-LOhwcwxw.png'></p>

Finally, PDP would only plot out the average predictions.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*HD_f_SjrQ9za2uLVGi61BQ.png'></p>

### ICEbox

Authors of ICEbox pointed out that Friedman’s PDP might obfuscate the complexity
of the model relationship since it is an average curve. The major idea of ICEbox
is to disaggregate the average plot by displaying the estimated functional
relationship for each observation. So instead of only plotting out the average
predictions, ICEbox displays all individual lines. (three lines in total in this
case)
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*lAudP4NYzJ_ZmsP_KSG7cA.png'></p>

They believe that at least in this way, with everything displayed in its initial
state, any interesting discovers wouldn’t be simply shielded from view because
of the averaging inherent in the PDP. Here is an vivid example showed in the
paper:
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*HGaFBFRpfEtWBxEKKxk_Ig.png'></p>

In this example, if you only look at the PDP in Figure b, you would think that
on average, X2 is not meaningfully associated with the predicted Y. While in
light of the scatter plot showed in Figure a, the conclusion is plainly wrong.
However, if you try to plot out the individual estimated conditional expectation
curves, everything is obvious.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*PwcYTruAXyP4PKioxpesig.png'></p>

The authors argue that, ICE algorithm gives the user insight into the several
variants of conditional relationships estimated by the black box.

## PDPbox

PDPbox aims to enrich the ideas mentioned by ICEbox in Python.

### **Improvement highlight**

1.  **Support **[one-hot](https://en.wikipedia.org/wiki/One-hot)** encoding
features.** For one-hot encoding feature, it is meaningless to investigate only
one predictor. Instead, all one-hot encoding predictors related to the original
feature should be investigated together. Take titanic data set as an example,
there is a categorical feature named ‘Embark’ and it contains three unique
values (C, S, Q). Since the majority of machine learning algorithms only accept
numeric features, categorical feature like ‘Embark’ is usually processed with
one-hot encoding and then turned into three new features: ‘Embark_C’, ‘Embark_S’
and ‘Embark_Q’. Therefore, if we want to investigate how ‘Embark’ feature would
influence the model prediction, we should investigate ‘Embark_C’, ‘Embark_S’ and
‘Embark_Q’ together.
1.  **For numeric features, create grids with percentile points.** ICEbox creates
grid points by selecting certain number of different values uniformly out of all
unique values of a certain feature. Well, when the number of required grid
points is larger than the number of unique values, it makes sense to use all
unique values as grid points because we don’t have other choice. However, when
the number of required grid points is smaller than the number of unique values,
we’d better choose the percentile points to make sure the grid points span
widely across the whole value range. The downside is that we might not be able
to get as many grid points as required because some percentile points might be
the same. (10 percentile point and 20 percentile point might be the same value)
1.  **Directly handle multiclass classifier.** In terms of classification problem,
PDPbox provides the same interface for both binary classifier and multiclass
classifier. Everything is handled automatically inside the function.
1.  **Support two variable interaction plot, **which is not yet supported in ICEbox
and PyCEbox.

### Structure

PDPbox has a simple structure as follows.

```python
class pdp_isolate_obj
# class for storing all useful information of single variable plot

def pdp_isolate(...)
# method for calculating all useful information of single variable plot

def pdp_plot(...)
# method for plotting out pdp plot for a single variable

class pdp_interact_obj
# class for storing all useful information of interaction plot for two variables

def pdp_interact(...)
# method for calculating all useful information of interaction plot for two variables

def pdp_interact_plot(...)
# method for plotting out interaction plot for two variables
```

### Examples: single variable plot

For single variable plot, the basic plot is exactly Friedman’s PDP, but with
standard deviation outline.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*I2VtqdOo-TbUhIEq_V1tQw.png'></p>

With basic plot, you can choose to plot out the original data points,
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*cMtF8Xcgf3HFmEKJX9Rb1Q.png'></p>

and also the individual estimated lines.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*s4qyzWTbOy35rTzjcBt9tA.png'></p>

You can also try to cluster the individual lines and only plot out the cluster
centers.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*5ZPkBl3L-etdlzwyfxUhUw.png'></p>

When it comes to multiclass problem, you can choose to plot out all classes at a
time,
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*dAHIyr66rlnKTUnmvZFfKg.png'></p>

or only plot out one class.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*c1WQVohwsRHL3m68QByopg.png'></p>

### Examples: two variable interaction plot

In terms of two variable interaction plot, the complete plot contains two
individual single variable plot and a two variable interaction contour plot. For
the individual single variable plot, all options are the same as those in single
variable plot.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*s9FaYPd201sF-UIqON3XCA.png'></p>

You can choose to only plot out the two variable interaction contour plot.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*7Q6uautib2rDTtCLKhiWow.png'></p>

When it comes to multiclass problem, you can choose to plot out all classes at a
time or just plot out one of them.
<p align='center'><img src='https://cdn-images-1.medium.com/max/800/1*NV2MexHn3cP4SXS5T3V00g.png'></p>

For more details of PDPbox, please refer to this [GitHub repo](https://github.com/SauceCat/PDPbox).
