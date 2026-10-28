"""Exploratory, prespecified sensitivity and training-label stress experiments."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_X_y
from .selectors import GBFRSSelector, build_balls, ball_dependency
from .secondary import make_secondary_selector

PROTOCOL='BUSBRA-ROBUSTNESS-20261028-V1'


def corrupt_training_labels(y, groups, fold, rate, seed=42):
    """Flip entire training groups; never receive or alter outer test observations."""
    if not 0 <= rate < .5:raise ValueError('Noise rate must be in [0,0.5)')
    y=np.asarray(y,dtype=int).copy();groups=np.asarray(groups)
    if set(y)!={0,1}:raise ValueError('Both training classes are required')
    unique=np.unique(groups)
    group_labels={g:np.unique(y[groups==g]) for g in unique}
    if any(len(labels)!=1 for labels in group_labels.values()):raise ValueError('Inconsistent group diagnosis')
    rng=np.random.RandomState(seed+fold); flipped=[]
    for label in (0,1):
        candidates=np.asarray([g for g in unique if group_labels[g][0]==label])
        count=int(np.floor(rate*len(candidates)))
        flipped.extend(rng.choice(candidates,count,replace=False).tolist())
    mask=np.isin(groups,flipped);y[mask]=1-y[mask]
    return y,{'requested_rate':rate,'seed':seed+fold,'training_groups':len(unique),
              'flipped_groups':len(flipped),'flipped_rows':int(mask.sum())}


def class_extreme_neighbor_dependency(centers, labels, weights, indices, normalizer):
    """Independent mathematical expression audit on fixed reconstruction balls.

    For each class c, r_i(c) is distance to the nearest ball not labeled c,
    including self for balls outside c. The criterion multiplies max_i r_i(c)
    by max_i weight(nearest_i(c)), sums over classes and normalizes by sqrt(|B|).
    This is an expression ablation, not a copied author implementation.
    """
    distances=cdist(centers[:,indices],centers[:,indices])
    score=0.
    for label in np.unique(labels):
        masked=np.where(labels[None,:]!=label,distances,np.inf)
        nearest=np.argmin(masked,axis=1)
        score+=float(masked.min(axis=1).max()*weights[nearest].max())
    return score/np.sqrt(normalizer)


class CriterionAblationSelector(GBFRSSelector):
    def __init__(self,purity=.9,random_state=42,tol=1e-12,criterion='own_ball'):
        super().__init__(purity=purity,random_state=random_state,tol=tol)
        self.criterion=criterion

    def fit(self,X,y):
        X,y=check_X_y(X,y,dtype=float)
        if set(y)!={0,1} or self.criterion not in ('own_ball','class_extreme_neighbor_weight'):
            raise ValueError('Invalid criterion or binary labels')
        self.n_features_in_=X.shape[1]
        span=np.ptp(X,axis=0);span[span==0]=1
        geometry=(X-X.min(axis=0))/span
        balls=build_balls(geometry,y,purity=self.purity,random_state=self.random_state)
        self.ball_sizes_=np.asarray([len(ball.indices) for ball in balls])
        self.ball_labels_=np.asarray([ball.label for ball in balls])
        centers=np.asarray([ball.center for ball in balls]);weights=self.ball_sizes_/len(X)
        dependency=ball_dependency if self.criterion=='own_ball' else class_extreme_neighbor_dependency
        chosen=[];remaining=list(range(X.shape[1]));history=[];previous=0.
        while remaining:
            scores=[dependency(centers,self.ball_labels_,weights,chosen+[a],len(chosen)+1) for a in remaining]
            best=int(np.argmax(scores))
            if scores[best]<=previous+self.tol:break
            chosen.append(remaining.pop(best));previous=scores[best];history.append(previous)
        if not chosen:raise ValueError('No improving feature in controlled ablation')
        self.selected_indices_=np.asarray(chosen,dtype=int);self.dependency_history_=history
        return self


@dataclass(frozen=True)
class Experiment:
    name:str
    method:str
    rate:float=0.
    purity:float=.9
    seed:int=42
    criterion:str='own_ball'

    def selector(self,method,config,author_path=None):
        if method=='gbfrs':
            if self.criterion!='own_ball':
                return CriterionAblationSelector(purity=self.purity,random_state=self.seed,criterion=self.criterion)
            return GBFRSSelector(purity=self.purity,random_state=self.seed)
        return make_secondary_selector(method,config,author_path)

    def labels(self,y,groups,fold):
        return corrupt_training_labels(y,groups,fold,self.rate)


def experiments():
    noise=[Experiment(f'{m}_noise_{int(rate*100)}',m,rate=rate)
           for rate in (.1,.2) for m in ('baseline','fuzzy_rough','gbfrs')]
    sensitivity=[Experiment(f'gbfrs_purity_{int(p*100)}','gbfrs',purity=p) for p in (.8,.95)]
    sensitivity+=[Experiment(f'gbfrs_seed_{seed}','gbfrs',seed=seed) for seed in (43,44)]
    return noise+sensitivity+[Experiment('gbfrs_criterion_ablation','gbfrs',criterion='class_extreme_neighbor_weight')]
