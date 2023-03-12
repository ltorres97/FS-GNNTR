from pandas import DataFrame
from matplotlib import pyplot
from scipy.stats import normaltest, ttest_ind, mannwhitneyu

####### Statistical Significance Analysis of Experimental Results #######

#FS-GNNTR
result1 = [0.7266, 0.7311, 0.7309, 0.7318, 0.7309, 0.7325, 0.7293, 0.7315, 0.7284, 0.7312, 0.7327, 0.724, 0.7319, 0.7313, 0.7298, 0.7354, 0.7316, 0.728, 0.7312, 0.7284, 0.7284, 0.7304, 0.727, 0.7228, 0.7273, 0.7328, 0.73, 0.729, 0.7315, 0.7316]

#Baseline
result2 = [0.5675, 0.6404, 0.5957, 0.6542, 0.6344, 0.5698, 0.5823, 0.5763, 0.5461, 0.5592, 0.636, 0.6475, 0.5633, 0.6457, 0.5843, 0.5605, 0.5857, 0.5829, 0.6076, 0.5931, 0.6279, 0.6496, 0.6123, 0.6092, 0.6029, 0.6556, 0.5949, 0.577, 0.5723, 0.5728]

results = DataFrame()
results['A'] = result1
results['B'] = result2

# Descriptive stats
print(results.describe())

# Box and whisker plot
results.boxplot()
pyplot.show()

# Histogram
results.hist()
pyplot.show()

#Normality test
value1, p1 = normaltest(result1)
value2, p2 = normaltest(result2)
print(p1, value1)
print(p2, value2)

gaussian = True

if p1 >= 0.05:
    print('It is likely that result1 is normal')
else:
    print('It is unlikely that result1 is normal')
    gaussian = False
 
if p2 >= 0.05:
    print('It is likely that result2 is normal')
else:
    print('It is unlikely that result2 is normal')
    gaussian = False
  
# Significance Test - Welch’s t-test - difference in variance - Parametric (Gaussian)

if gaussian == True:
    value, pvalue = ttest_ind(result1, result2, equal_var=False)
    print(value, pvalue)
    if pvalue > 0.05:
        print('Samples are likely drawn from the same distributions (fail to reject H0)')
    else:
        print('Samples are likely drawn from different distributions (reject H0)')
        
# Significance Test - Mann-Whitney - Non-gaussian distributions - Non-parametric

if gaussian == False:
    value, pvalue =  mannwhitneyu(result1, result2)
    print(value, pvalue)
    if pvalue > 0.05:
     print('Samples are likely drawn from the same distributions (fail to reject H0)')
    else:
     print('Samples are likely drawn from different distributions (reject H0)')

