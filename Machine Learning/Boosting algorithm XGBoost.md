# Boosting algorithm: XGBoost

This article continues the previous post [Boosting algorithm:
GBM](https://medium.com/towards-data-science/boosting-algorithm-gbm-97737c63daa3).
This time we are going to discuss XGBoost! (Finally!)

## XGBoost: Extreme Gradient Boosting

XGBoost, short for “Extreme Gradient Boosting”, was introduced by Chen in 2014.
Since its introduction, XGBoost has become one of the most popular machine
learning algorithm. In this post, we will dive deeply into the algorithm itself
and try to figure out how XGBoost differs from the traditional boosting
algorithms GBM.

As mentioned in the previous post, GBM divides the optimization problem into two
parts by first determining the direction of the step and then optimizing the
step length. Different from GBM, XGBoost tries to determine the step directly by
solving
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*QAIReONJ8r6AmHKuaVotdQ.png'>
</p>

for each **x** in the data set. By doing second-order Taylor expansion of the
loss function around the current estimate **f(m-1)(x)**, we get
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*WWInCZCh7KhQmi8nscJMYw.png'>
</p>

where **g_m(x)** is the gradient, same as the one in GBM, and **h_m(x)** is the
Hessian (second order derivative) at the current estimate:
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*qYj8oeFqvmAc5X8a66C7uQ.png'>
</p>

Then the loss function can be rewritten as
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*QL-uJ9zBKrT19ugCYrbO4A.png'>
</p>

Letting **G_jm** represents the sum of gradient in region **j** and **H_jm**
equals to the sum of hessian in region **j**, the equation can be rewritten as
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*RE57Lvsbure_KICFBifvtA.png'>
</p>

With the fixed learned structure, for each region, it is straightforward to
determine the optimal weight :
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*qGF58wI3LzNYEzU9U2VEdw.png'>
</p>

Plugging it back to the loss function, we get
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*QgPYkdcmTz1Q7SNPsOOlWQ.png'>
</p>

According to Chen, this is the structure score for a tree. The smaller the score
is, the better the structure is. Thus, for each split to make, the proxy gain is
defined as
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*vCRDHfATnfhsJXwN8ZX89w.png'>
</p>

Well, all deductions above didn’t take regularization into consideration. Note
that XGBoost provides variety of regularization to improve generalization
performance. Taking regularization into consideration, we can rewrite the loss
function as
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*ri73yWvUMiaFW5-hDnFu1A.png'>
</p>

where **γ** is the penalization term on the number of terminal nodes, **α** and
**λ** are for **L1** and **L2** regularization respectively. The optimal weight
for each region **j** is calculated as:
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*pgAVG1tCOZrDbO5_3NDn0Q.png'>
</p>

The gain of each split is defined correspondingly:
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*J_VT3VcoKLL-yFRV9eBgAg.png'>
</p>

## Walk through XGBoost source code

For better understanding, we are going to walk through the source code of
[XGBoost](https://github.com/dmlc/xgboost/tree/master/src). For simplification,
we will only focus on binary classification and the most important code
snippets.

### Update one boost round
```cpp
void UpdateOneIter(int iter, DMatrix* train) override {
  this->LazyInitDMatrix(train);
  this->PredictRaw(train, &preds_);
  obj_->GetGradient(preds_, train->info(), iter, &gpair_);
  gbm_->DoBoost(train, &gpair_, obj_.get());
}
```

### Get gradient information for each instance
```cpp
void GetGradient(const std::vector<bst_float> &preds,
                 const MetaInfo &info,
                 int iter, std::vector<bst_gpair> *out_gpair) override {
  // start calculating gradient
  const omp_ulong ndata = static_cast<omp_ulong>(preds.size());
  for (omp_ulong i = 0; i < ndata; ++i) {
    bst_float p = Loss::PredTransform(preds[i]);
    bst_float w = info.GetWeight(i);
    if (info.labels[i] == 1.0f) w *= param_.scale_pos_weight;
    if (!Loss::CheckLabel(info.labels[i])) label_correct = false;
    out_gpair->at(i) = bst_gpair(Loss::FirstOrderGradient(p, info.labels[i]) * w,
                                 Loss::SecondOrderGradient(p, info.labels[i]) * w);
  }
}
```

### Loss function for binary classification task
```cpp
XGBOOST_REGISTER_OBJECTIVE(LogisticClassification, "binary:logistic")
.describe("Logistic regression for binary classification task.")
.set_body([]() { return new RegLossObj<LogisticClassification>(); });

// logistic loss for probability regression task
struct LogisticRegression {
  static bst_float PredTransform(bst_float x) { return common::Sigmoid(x); }
  
  // the objective function should provide the first order gradient and the second order gradient
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) { return predt - label; }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const float eps = 1e-16f;
    return std::max(predt * (1.0f - predt), eps);
  }
};

// logistic loss for binary classification task.
struct LogisticClassification : public LogisticRegression {
  static const char* DefaultEvalMetric() { return "error"; }
};
```

### DoBoost with updaters
```cpp
void DoBoost(DMatrix* p_fmat, std::vector<bst_gpair>* in_gpair,
             ObjFunction* obj) override {
  const std::vector<bst_gpair>& gpair = *in_gpair;
  std::vector<std::vector<std::unique_ptr<RegTree> > > new_trees;
  // for binary classification task
  if (mparam.num_output_group == 1) {
    std::vector<std::unique_ptr<RegTree> > ret;
    BoostNewTrees(gpair, p_fmat, 0, &ret);
    new_trees.push_back(std::move(ret));
  } else {
    // others
  }
}

// do group specific group
inline void BoostNewTrees(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat, 
                          int bst_group, std::vector<std::unique_ptr<RegTree> >* ret) {
  this->InitUpdater();
  std::vector<RegTree*> new_trees;
  // create the trees
  // for boosting, num_parallel_tree equals to 1
  for (int i = 0; i < tparam.num_parallel_tree; ++i) {
    if (tparam.process_type == kDefault) {
      // create new tree
      std::unique_ptr<RegTree> ptr(new RegTree());
      ptr->param.InitAllowUnknown(this->cfg);
      ptr->InitModel();
      new_trees.push_back(ptr.get());
      ret->push_back(std::move(ptr));
    } else if (tparam.process_type == kUpdate) {
      // update the existing tree
    }
  }
  // update the trees
  for (auto& up : updaters) {
    up->Update(gpair, p_fmat, new_trees);
  }
}
```

### Updater initialization
```cpp
// initialize updater before using them
inline void InitUpdater() {
  if (updaters.size() != 0) return;
  // updater_seq is the string defining the sequence of tree updaters
  // default is set as grow_colmaker,prune
  std::string tval = tparam.updater_seq;
  std::vector<std::string> ups = common::Split(tval, ',');
  for (const std::string& pstr : ups) {
    std::unique_ptr<TreeUpdater> up(TreeUpdater::Create(pstr.c_str()));
    up->Init(this->cfg);
    updaters.push_back(std::move(up));
  }
}
```

**updater_seq** is a comma separated string defining the sequence of tree
updaters to run, providing a modular way to construct and to modify the trees.
In default, it is set as “**grow_colmaker,prune**”, which means first run
**updater_colmaker** and then run **updater_prune**.

For **updater_colmaker**, each tree is updated by the **builder** depth by
depth.
```cpp
// inside ColMaker
void Update(const std::vector<bst_gpair> &gpair, DMatrix* dmat, 
            const std::vector<RegTree*> &trees) override {
  // build tree
  for (size_t i = 0; i < trees.size(); ++i) {
    Builder builder(param);
    builder.Update(gpair, dmat, trees[i]);
  }
}

// Update method of builder
virtual void Update(const std::vector<bst_gpair>& gpair, DMatrix* p_fmat, RegTree* p_tree) {
  this->InitData(gpair, *p_fmat, *p_tree);
  this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
  for (int depth = 0; depth < param.max_depth; ++depth) {
    this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
    this->ResetPosition(qexpand_, p_fmat, *p_tree);
    this->UpdateQueueExpand(*p_tree, &qexpand_);
    this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
    // if nothing left to be expand, break
    if (qexpand_.size() == 0) break;
  }
}
```

For **updater_prune**, tree leaves are pruned recursively.
```cpp
/*! \brief do pruning of a tree */
inline void DoPrune(RegTree &tree) { // NOLINT(*)
  int npruned = 0;
  for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
    if (tree[nid].is_leaf()) {
      npruned = this->TryPruneLeaf(tree, nid, tree.GetDepth(nid), npruned);
    }
  }
}

// try to prune off current leaf
inline int TryPruneLeaf(RegTree &tree, int nid, int depth, int npruned) { // NOLINT(*)
  if (s.leaf_child_cnt >= 2 && param.need_prune(s.loss_chg, depth - 1)) {
    // need to be pruned
    tree.ChangeToLeaf(pid, param.learning_rate * s.base_weight);
    // tail recursion
    return this->TryPruneLeaf(tree, pid, depth - 1, npruned + 2);
  } else {
    return npruned;
  }
}

/*! \brief given the loss change, whether we need to invoke pruning */
inline bool need_prune(double loss_chg, int depth) const {
  return loss_chg < this->min_split_loss;
}
```
Calculations for several important statistics are listed as follow.

### Loss change for each split
```cpp
loss_chg = static_cast<bst_float>(constraints_[nid].CalcSplitGain(param, fid, e.stats, c) - snode[nid].root_gain);
snode[nid].root_gain = static_cast<float>(constraints_[nid].CalcGain(param, snode[nid].stats));

inline double CalcSplitGain(const TrainParam &param, bst_uint split_index,
                            GradStats left, GradStats right) const {
  return left.CalcGain(param) + right.CalcGain(param);
}
```

### Gain calculation for each tree node
```cpp
template <typename TrainingParams, typename T>
XGB_DEVICE inline T CalcGain(const TrainingParams &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight)
    return 0.0;
  if (p.max_delta_step == 0.0f) {
    if (p.reg_alpha == 0.0f) {
      return Sqr(sum_grad) / (sum_hess + p.reg_lambda);
    } else {
      return Sqr(ThresholdL1(sum_grad, p.reg_alpha)) /
             (sum_hess + p.reg_lambda);
    }
  } else {
    T w = CalcWeight(p, sum_grad, sum_hess);
    T ret = sum_grad * w + 0.5 * (sum_hess + p.reg_lambda) * Sqr(w);
    if (p.reg_alpha == 0.0f) {
      return -2.0 * ret;
    } else {
      return -2.0 * (ret + p.reg_alpha * std::abs(w));
    }
  }
}
```

### Weight calculation for each tree node
```cpp
// calculate weight given the statistics
template <typename TrainingParams, typename T>
XGB_DEVICE inline T CalcWeight(const TrainingParams &p, T sum_grad,
                               T sum_hess) {
  if (sum_hess < p.min_child_weight)
    return 0.0;
  T dw;
  if (p.reg_alpha == 0.0f) {
    dw = -sum_grad / (sum_hess + p.reg_lambda);
  } else {
    dw = -ThresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda);
  }
  if (p.max_delta_step != 0.0f) {
    if (dw > p.max_delta_step)
      dw = p.max_delta_step;
    if (dw < -p.max_delta_step)
      dw = -p.max_delta_step;
  }
  return dw;
}
```

To predict a new instance, first get the leaf indexes, then sum up the leaf
values.
```cpp
// predict the leaf scores without dropped trees
inline bst_float PredValue(const RowBatch::Inst &inst,
                           int bst_group,
                           unsigned root_index,
                           RegTree::FVec *p_feats,
                           unsigned tree_begin,
                           unsigned tree_end) {
  bst_float psum = 0.0f;
  p_feats->Fill(inst);
  for (size_t i = tree_begin; i < tree_end; ++i) {
    if (tree_info[i] == bst_group) {
      bool drop = (std::binary_search(idx_drop.begin(), idx_drop.end(), i));
      if (!drop) {
        int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
        psum += weight_drop[i] * (*trees[i])[tid].leaf_value();
      }
    }
  }
  p_feats->Drop(inst);
  return psum;
}
```

The leaf value is calculated as:
```cpp
inline void Refresh(const TStats *gstats,
                    int nid, RegTree *p_tree) {
  if (tree[nid].is_leaf()) {
    if (param.refresh_leaf) {
      tree[nid].set_leaf(tree.stat(nid).base_weight * param.learning_rate);
    }
  }
}
```

## Compare GBM and XGBoost

**GBM has broader application.** At each iteration, both GBM and XGBoost need to
calculate gradient at current estimate. XGBoost also needs to calculate hessian,
requiring the objective function to be twice differentiable (strictly convex).
GBM only requires a differentiable loss function, thus it can be used in more
applications.

**XGBoost is faster.** Comparing the weights calculated by GBM and XGBoost, for
GBM, the weight is simply the average value of the gradients, while for XGBoost,
it is the sum of gradients scaled by the sum of hessians.
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*QPZl-djU1xI8tuW8up9Pew.png'>
</p>

For XGBoost, the weight is also known as the [Newton
“step”](https://en.wikipedia.org/wiki/Newton's_method_in_optimization), which
naturally has step length of 1. Thus, line search is not necessary for XGBoost.
This might be the reason why XGBoost is always much faster than GBM.

**XGBoost provides more regularization options**, including L1(**α**) and
L2(**λ**) regularization as well as penalization on the number of leaf
nodes(**γ**).
<p align="center">
<img src='https://cdn-images-1.medium.com/max/800/1*hdtQ3c_6LhTw1Kal5Gi9Uw.png'>
</p>

However, in terms of GBM in sklearn package, various useful regularization
strategies are also provided. In version 0.19, parameter
[min_impurity_decrease](https://github.com/scikit-learn/scikit-learn/pull/8449),
similar to **γ** in XGBoost, is added.

>All tree based estimators now accept a `min_impurity_decrease` parameter in lieu of the `min_impurity_split`, which is now deprecated. The `min_impurity_decrease` helps stop splitting the nodes in which the weighted impurity decrease from splitting is no longer at least `min_impurity_decrease`.

Besides, for each tree in the ensemble, regularization options, including `min_samples_split` , `min_samples_leaf` , `min_weight_fraction_leaf` and `max_leaf_nodes`, are implemented. For XGBoost, individual tree is regularized by `max_depth`, `min_child_weight`, `max_delta_step` as well as L1 and L2 penalization.

**XGBoost introduces more randomization.** For GBM in sklearn package, we have parameter `subsample` (similar to subsample in XGBoost) to implement Stochastic Gradient Boosting, and `max_features` for column sampling. XGBoost also provides two similar options. The only difference is that XGBoost provides two levels of column sampling, `colsample_bytree` and `colsample_bylevel`, thus introducing more randomness into the learning process.
