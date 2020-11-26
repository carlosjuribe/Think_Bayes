'''
By: Carlos Jiménez Uribe
My personal version of the classes utilized in the original module of the author, "thinkbayes.py"
'''
from thinkbayes import *  # just take all to avoid errors
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

#######################  classes  #######################
class Suite(Pmf):
    """
    Represents a suite of hypotheses and their probabilities (as PMFs).
    To be inherited by another class for a specicif problem, where the Likelihood
    method needs to be specified.
    """

    def Update(self, data):
        """Updates each hypothesis based on the data.

        data: any representation of the data

        returns: the normalizing constant
        """
        for hypo in list(self.Values()):
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        return self.Normalize()
        
    def Print(self):
        """Prints the hypotheses and their probabilities."""
        print(f"Distribution of probabilites now: \n{'---'*15}\n")
        for hypo, prob in sorted(self.Items()):
            print(f"Hypothesis {hypo}: {prob:.4f}")
            
    def PlotProb(self):
        """ Plot the current probability distribution """
        fig, ax = plt.subplots()
        ax.plot(self.GetDict().keys(), self.GetDict().values(), label='pmf')
        ax.autoscale(tight=True)
        ax.locator_params(nbins=3)    
        ax.set_xlabel('hypotheses', fontsize=12)    
        ax.set_ylabel('probability', fontsize=12)    
        ax.set_title('Current probability distribution', fontsize=12)
        plt.legend()
        return ax
    
    def PlotMean(self):
        """ Plots the mean of the distribution on top of it, and returns its value """
        ax = self.PlotProb()
        self.MeanProb = self.Mean()
        ax.set_title(f"Mean: {self.MeanProb:.4f}")
        ax.axvline(self.MeanProb, color='red', label='mean')
        plt.legend()
            

class Train(Suite):
    """ Suite for dealing with distribution for the Locomotive Problem """
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0.0  # impossible to see a train with number i if there are fewer than i trains
        else:
            return 1.0 / hypo  # Likelihood to find train i in a population of k trains (k > i)
        
    def PlotProb(self):
        """ Plot the current probability distribution """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.GetDict().keys(), self.GetDict().values(), label='pmf')
        ax.autoscale(tight=True)
        ax.locator_params(nbins=3)    
        ax.set_xlabel('hypotheses', fontsize=12)    
        ax.set_ylabel('probability', fontsize=12)    
        ax.set_title('Current probability distribution', fontsize=12)
        plt.legend()
        return ax
    
    def PlotMean(self):
        """ Plots the mean of the distribution on top of it, and returns its value """
        ax = self.PlotProb()
        self.MeanProb = self.Mean()
        ax.set_title(f"Mean: {self.MeanProb:.4f}")
        ax.axvline(self.MeanProb, color='red', label='mean')
        plt.legend()
        return self.MeanProb
    
    def GetMaxProb(self):
        """ Returns the maximum value of all the distribution """
        return max(self.GetDict().values())


class TrainPowerLawPrior(Train):  # we inherit from our previous class, just to change the init
    """ 
    Suite for dealing with distribution for the Locomotive Problem, assuming a Power-Law prior
    """
    def __init__(self, hypos, alpha=1.0):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, hypo**(- alpha))  # assign a power-law probab distribution at first
        self.Normalize()


#######################  functions  #######################
def init_model_make_posterior(upper_bound: float, observed_data: float, constructor):
    """ 
    Makes and updates a Suite.
    :param upper_bound: maximum number of expected trains (possibilities)
    :param observed_data: new train number observed
    :param constructor: a class that contains a LikeLihood function, inherited from Suite
    """
    hypos = range(1, upper_bound + 1)
    suite = constructor(hypos)
    suite.name = f"upper_bound_{upper_bound}"
    
    for obs in observed_data:
        suite.Update(obs)
    return suite


def compare_posteriors_for_same_data(constructors: List, labels: List[str], upper_bound: float, 
                                     observed_data: List[float],):
    """ 
    Runs the analysis with two different priors and compares them 
    :param constructors: list of classes that contain a LikeLihood function, inherited from Suite
    :param labels: names for those constructors
    :param upper_bound: maximum number of expected trains (possibilities)
    :param observed_data: list of new train numbers observed
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for constructor, label in zip(constructors, labels):
        suite = init_model_make_posterior(upper_bound, observed_data, constructor)
        suite.name = label
        suite_dict = suite.GetDict()
        ax.plot(suite_dict.keys(), suite_dict.values(), label=label)
        ax.autoscale(tight=True, axis='both')
        ax.annotate(f"New datapoints observed: {observed_data}", (600, 0.01) )
        ax.set_title("Comparison of posteriors for different priors and same data")
        ax.set_xlabel("Possible values"); ax.set_ylabel("Probability")
        ax.legend(fontsize=11)
        

def percentile(pmf, percentage: float):
    """ Computes the 'percentage' percentile of the PMF distribution """
    total = 0
    for val, prob in pmf.Items():
        total += prob
        if total >= percentage / 100.0:
            return val


def credible_interval(pmf, percentage: float = 90.0):
    """ Computes the 'percentage'% credible interval for the given distribution """
    tail = (100 - percentage) / 2
    return percentile(pmf, tail), percentile(pmf, 100 - tail)


def plot_distrib_with_CI(pmf, CI_level: float = 90.0, title=""):
    """ Plots the credible interval specified on top of the distribution """
    mean = pmf.MeanProb
    CI = credible_interval(pmf, percentage=CI_level)
    ax = pmf.PlotProb()
    ax.set_title(f"Current Distribution with {CI_level}% credible Intervals\n{title}")
    ax.axvline(mean, color='red', label='mean')
    ax.axvspan(CI[0], CI[1], alpha=0.3, color='red', label=f"{CI_level}% CI")
    ax.annotate(f"Mean: {mean:.0f}", (750, 0.015), color='darkred')
    ax.legend(fontsize=11)
    

def compare_distributions(suites: List, labels=['prior1', 'prior2']):
    """
    Given a list of suites, plots each current distribution on the same plot, for comparison 
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for suite, label in zip(suites, labels):
        suite_dict = suite.GetDict()
        ax.plot(suite_dict.keys(), suite_dict.values(), label=label)
        ax.autoscale(tight=True, axis='x')
        ax.set_title("Comparison of distributions", fontsize=14)
        ax.set_xlabel("Possible values", fontsize=12); ax.set_ylabel("Probability (now)")
        ax.legend(fontsize=11)
        
        
def summary_distributions(suites: List, labels=['prior1', 'prior2']):
    """ Given a list of suites, prints out relevant point estimates for each suite """
    for suite, label in zip(suites, labels):
        mean = suite.Mean()
        median = percentile(suite, 50)
        CI_90 = credible_interval(suite, 90)
        print(f"\n{'---'*20}\nSummary for suite: {label}\n"
               f" - Mean: {mean:0.2f}\n - Median: {median}\n - 90%-CI: {CI_90}")
        
        

###################### from the original module ############################
# class Pmf(_DictWrapper):
#     """Represents a probability mass function.
    
#     Values can be any hashable type; probabilities are floating-point.
#     Pmfs are not necessarily normalized.
#     """

#     def Prob(self, x, default=0):
#         """Gets the probability associated with the value x.

#         Args:
#             x: number value
#             default: value to return if the key is not there

#         Returns:
#             float probability
#         """
#         return self.d.get(x, default)

#     def Probs(self, xs):
#         """Gets probabilities for a sequence of values."""
#         return [self.Prob(x) for x in xs]

#     def MakeCdf(self, name=None):
#         """Makes a Cdf."""
#         return MakeCdfFromPmf(self, name=name)

#     def ProbGreater(self, x):
#         """Probability that a sample from this Pmf exceeds x.

#         x: number

#         returns: float probability
#         """
#         t = [prob for (val, prob) in self.d.items() if val > x]
#         return sum(t)

#     def ProbLess(self, x):
#         """Probability that a sample from this Pmf is less than x.

#         x: number

#         returns: float probability
#         """
#         t = [prob for (val, prob) in self.d.items() if val < x]
#         return sum(t)

#     def __lt__(self, obj):
#         """Less than.

#         obj: number or _DictWrapper

#         returns: float probability
#         """
#         if isinstance(obj, _DictWrapper):
#             return PmfProbLess(self, obj)
#         else:
#             return self.ProbLess(obj)

#     def __gt__(self, obj):
#         """Greater than.

#         obj: number or _DictWrapper

#         returns: float probability
#         """
#         if isinstance(obj, _DictWrapper):
#             return PmfProbGreater(self, obj)
#         else:
#             return self.ProbGreater(obj)

#     def __ge__(self, obj):
#         """Greater than or equal.

#         obj: number or _DictWrapper

#         returns: float probability
#         """
#         return 1 - (self < obj)

#     def __le__(self, obj):
#         """Less than or equal.

#         obj: number or _DictWrapper

#         returns: float probability
#         """
#         return 1 - (self > obj)

#     def __eq__(self, obj):
#         """Equal to.

#         obj: number or _DictWrapper

#         returns: float probability
#         """
#         if isinstance(obj, _DictWrapper):
#             return PmfProbEqual(self, obj)
#         else:
#             return self.Prob(obj)

#     def __ne__(self, obj):
#         """Not equal to.

#         obj: number or _DictWrapper

#         returns: float probability
#         """
#         return 1 - (self == obj)

#     def Normalize(self, fraction=1.0):
#         """Normalizes this PMF so the sum of all probs is fraction.

#         Args:
#             fraction: what the total should be after normalization

#         Returns: the total probability before normalizing
#         """
#         if self.log:
#             raise ValueError("Pmf is under a log transform")

#         total = self.Total()
#         if total == 0.0:
#             raise ValueError('total probability is zero.')
#             logging.warning('Normalize: total probability is zero.')
#             return total

#         factor = float(fraction) / total
#         for x in self.d:
#             self.d[x] *= factor

#         return total

#     def Random(self):
#         """Chooses a random element from this PMF.

#         Returns:
#             float value from the Pmf
#         """
#         if len(self.d) == 0:
#             raise ValueError('Pmf contains no values.')

#         target = random.random()
#         total = 0.0
#         for x, p in self.d.items():
#             total += p
#             if total >= target:
#                 return x

#         # we shouldn't get here
#         assert False

#     def Mean(self):
#         """Computes the mean of a PMF.

#         Returns:
#             float mean
#         """
#         mu = 0.0
#         for x, p in self.d.items():
#             mu += p * x
#         return mu

#     def Var(self, mu=None):
#         """Computes the variance of a PMF.

#         Args:
#             mu: the point around which the variance is computed;
#                 if omitted, computes the mean

#         Returns:
#             float variance
#         """
#         if mu is None:
#             mu = self.Mean()

#         var = 0.0
#         for x, p in self.d.items():
#             var += p * (x - mu) ** 2
#         return var

#     def MaximumLikelihood(self):
#         """Returns the value with the highest probability.

#         Returns: float probability
#         """
#         prob, val = max((prob, val) for val, prob in self.Items())
#         return val

#     def CredibleInterval(self, percentage=90):
#         """Computes the central credible interval.

#         If percentage=90, computes the 90% CI.

#         Args:
#             percentage: float between 0 and 100

#         Returns:
#             sequence of two floats, low and high
#         """
#         cdf = self.MakeCdf()
#         return cdf.CredibleInterval(percentage)

#     def __add__(self, other):
#         """Computes the Pmf of the sum of values drawn from self and other.

#         other: another Pmf

#         returns: new Pmf
#         """
#         try:
#             return self.AddPmf(other)
#         except AttributeError:
#             return self.AddConstant(other)

#     def AddPmf(self, other):
#         """Computes the Pmf of the sum of values drawn from self and other.

#         other: another Pmf

#         returns: new Pmf
#         """
#         pmf = Pmf()
#         for v1, p1 in self.Items():
#             for v2, p2 in other.Items():
#                 pmf.Incr(v1 + v2, p1 * p2)
#         return pmf

#     def AddConstant(self, other):
#         """Computes the Pmf of the sum a constant and  values from self.

#         other: a number

#         returns: new Pmf
#         """
#         pmf = Pmf()
#         for v1, p1 in self.Items():
#             pmf.Set(v1 + other, p1)
#         return pmf

#     def __sub__(self, other):
#         """Computes the Pmf of the diff of values drawn from self and other.

#         other: another Pmf

#         returns: new Pmf
#         """
#         pmf = Pmf()
#         for v1, p1 in self.Items():
#             for v2, p2 in other.Items():
#                 pmf.Incr(v1 - v2, p1 * p2)
#         return pmf

#     def Max(self, k):
#         """Computes the CDF of the maximum of k selections from this dist.

#         k: int

#         returns: new Cdf
#         """
#         cdf = self.MakeCdf()
#         cdf.ps = [p ** k for p in cdf.ps]
#         return cdf

#     def __hash__(self):
#         # FIXME
#         # This imitates python2 implicit behaviour, which was removed in python3

#         # Some problems with an id based hash:
#         # looking up different pmfs with the same contents will give different values
#         # looking up a new Pmf will always produce a keyerror

#         # A solution might be to make a "FrozenPmf" immutable class (like frozenset)
#         # and base a hash on a tuple of the items of self.d
#         return id(self)

# class _DictWrapper(object):
#     """An object that contains a dictionary."""

#     def __init__(self, values=None, name=''):
#         """Initializes the distribution.

#         hypos: sequence of hypotheses
#         """
#         self.name = name
#         self.d = {}

#         # flag whether the distribution is under a log transform
#         self.log = False

#         if values is None:
#             return

#         init_methods = [
#             self.InitPmf,
#             self.InitMapping,
#             self.InitSequence,
#             self.InitFailure,
#             ]

#         for method in init_methods:
#             try:
#                 method(values)
#                 break
#             except AttributeError:
#                 continue

#         if len(self) > 0:
#             self.Normalize()

#     def InitSequence(self, values):
#         """Initializes with a sequence of equally-likely values.

#         values: sequence of values
#         """
#         for value in values:
#             self.Set(value, 1)

#     def InitMapping(self, values):
#         """Initializes with a map from value to probability.

#         values: map from value to probability
#         """
#         for value, prob in values.items():
#             self.Set(value, prob)

#     def InitPmf(self, values):
#         """Initializes with a Pmf.

#         values: Pmf object
#         """
#         for value, prob in values.Items():
#             self.Set(value, prob)

#     def InitFailure(self, values):
#         """Raises an error."""
#         raise ValueError('None of the initialization methods worked.')

#     def __len__(self):
#         return len(self.d)

#     def __iter__(self):
#         return iter(self.d)

#     def keys(self):
#         return iter(self.d)

#     def __contains__(self, value):
#         return value in self.d

#     def Copy(self, name=None):
#         """Returns a copy.

#         Make a shallow copy of d.  If you want a deep copy of d,
#         use copy.deepcopy on the whole object.

#         Args:
#             name: string name for the new Hist
#         """
#         new = copy.copy(self)
#         new.d = copy.copy(self.d)
#         new.name = name if name is not None else self.name
#         return new

#     def Scale(self, factor):
#         """Multiplies the values by a factor.

#         factor: what to multiply by

#         Returns: new object
#         """
#         new = self.Copy()
#         new.d.clear()

#         for val, prob in self.Items():
#             new.Set(val * factor, prob)
#         return new

#     def Log(self, m=None):
#         """Log transforms the probabilities.
        
#         Removes values with probability 0.

#         Normalizes so that the largest logprob is 0.
#         """
#         if self.log:
#             raise ValueError("Pmf/Hist already under a log transform")
#         self.log = True

#         if m is None:
#             m = self.MaxLike()

#         for x, p in self.d.items():
#             if p:
#                 self.Set(x, math.log(p / m))
#             else:
#                 self.Remove(x)

#     def Exp(self, m=None):
#         """Exponentiates the probabilities.

#         m: how much to shift the ps before exponentiating

#         If m is None, normalizes so that the largest prob is 1.
#         """
#         if not self.log:
#             raise ValueError("Pmf/Hist not under a log transform")
#         self.log = False

#         if m is None:
#             m = self.MaxLike()

#         for x, p in self.d.items():
#             self.Set(x, math.exp(p - m))

#     def GetDict(self):
#         """Gets the dictionary."""
#         return self.d

#     def SetDict(self, d):
#         """Sets the dictionary."""
#         self.d = d

#     def Values(self):
#         """Gets an unsorted sequence of values.

#         Note: one source of confusion is that the keys of this
#         dictionary are the values of the Hist/Pmf, and the
#         values of the dictionary are frequencies/probabilities.
#         """
#         return self.d.keys()

#     def Items(self):
#         """Gets an unsorted sequence of (value, freq/prob) pairs."""
#         return self.d.items()

#     def Render(self):
#         """Generates a sequence of points suitable for plotting.

#         Returns:
#             tuple of (sorted value sequence, freq/prob sequence)
#         """
#         return zip(*sorted(self.Items()))

#     def Print(self):
#         """Prints the values and freqs/probs in ascending order."""
#         for val, prob in sorted(self.d.items()):
#             print(val, prob)

#     def Set(self, x, y=0):
#         """Sets the freq/prob associated with the value x.

#         Args:
#             x: number value
#             y: number freq or prob
#         """
#         self.d[x] = y

#     def Incr(self, x, term=1):
#         """Increments the freq/prob associated with the value x.

#         Args:
#             x: number value
#             term: how much to increment by
#         """
#         self.d[x] = self.d.get(x, 0) + term

#     def Mult(self, x, factor):
#         """Scales the freq/prob associated with the value x.

#         Args:
#             x: number value
#             factor: how much to multiply by
#         """
#         self.d[x] = self.d.get(x, 0) * factor

#     def Remove(self, x):
#         """Removes a value.

#         Throws an exception if the value is not there.

#         Args:
#             x: value to remove
#         """
#         del self.d[x]

#     def Total(self):
#         """Returns the total of the frequencies/probabilities in the map."""
#         total = sum(self.d.values())
#         return total

#     def MaxLike(self):
#         """Returns the largest frequency/probability in the map."""
#         return max(self.d.values())