import pandas as pd
from numpy import zeros
import numpy as np
import portfolioopt as pfopt

def view_link(etf_list, views):
    n = len(etf_list) # Number of ETFs
    m = len(views) # Number of Views
    # View Matrix
    view_m = [views[i][2] for i in views]
    view_m = pd.DataFrame({'Views':view_m})
    # Link Matrix
    link_m = zeros([m,n]) # Create Link Matrix Filled with 0
    i = 0 # View Record
    for each in views.values():
        num_etf = len(each[0])
        for j in range(num_etf):
            link_m[i][etf_list.index(each[0][j])] = each[1][j]
        i += 1
    link_m = pd.DataFrame(link_m, index = list(views.keys()), columns = etf_list)
    return(view_m, link_m)

## preprocess daily returns of previous month ##
dailyreturn = pd.read_csv('returns_17_10.csv')
data = dailyreturn
d = {}
for i in data.columns.values:
    dval = data[i].tolist()
    d[i] = dval
# d is the dictionary we can use to put into the blm model
stk_list = list(d.keys())
data = list(d.values())
n = len(data[0])
cov_m = pd.DataFrame(np.cov(data))
rtn_m = pd.Series([np.mean(each) for each in data])

#tangency_portfolio(pandas.DataFrame, pandas.Series)
w_neutral = pfopt.tangency_portfolio(cov_m, rtn_m, allow_short = True)
w_neutral = pd.DataFrame([round(each,4) for each in w_neutral],
                         index = stk_list, columns = ['Weights'])

### The asolute views obtained from view prediction AI-model						 
##############
view = {}
view['view1'] = (['LQD'],[1],0.003507782) 
view['view2'] = (['SPY'],[1],0.000190048) 
view['view3'] = (['VOT'],[1],0.007558567)
view['view4'] = (['IWV'],[1],0.001878394) 
view['view5'] = (['IWP'],[1],0.004234864) 
view['view6'] = (['IJK'],[1],0.019953683) 
view['view7'] = (['IVV'],[1],0.01048204)
##############

p,q = view_link(stk_list, view) # p: View Matrix   q: Link Matrix

delta = 2.65 # Risk Aversion
tau = 0.01 # Scaling Factor
q = q.as_matrix()
p = list(p['Views'])
omega = tau * (q.dot(cov_m).dot(q.transpose()))
omega = np.diag(np.diag(omega, k = 0))

adj_rtn = rtn_m+tau*cov_m.dot(q.transpose()).dot(
    np.linalg.inv(omega+tau*(q.dot(cov_m).dot(
        q.transpose())))).dot(p-q.dot(rtn_m))

# new covariance matrix for BLM
tauV = tau*cov_m
mid_eq = np.linalg.inv(np.dot(np.dot(q,tauV),q.T) + omega)
adj_cov = cov_m + tauV - tauV.dot(q.T).dot(mid_eq).dot(q).dot(tauV)

w_expressing = pfopt.tangency_portfolio(adj_cov, adj_rtn, allow_short = True)
##w_expressing = np.linalg.inv(delta*cov_m).dot(rtn_m)
w_expressing = pd.DataFrame([round(each,4) for each in w_expressing],
                            index = stk_list, columns = ['Weights'])
print("Weights with Expressing Views:") 
print(w_expressing)