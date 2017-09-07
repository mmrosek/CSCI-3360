from sklearn.datasets import load_svmlight_file
import time

filepath = "/Users/Miller/Documents/CSCI 3360/a9a.txt"

def load_libsvm(filepath):

    file = open(filepath, 'r')
    list_of_rows = file.readlines()

    # Splits up each line using " " as delimiter
    line_splits = [list_of_rows[i].split() for i in range(len(list_of_rows))]
  
    target = np.array([int(line_splits[x][0].strip("+")) for x in range(len(line_splits))])
    
    indices = []
    for i in range(len(line_splits)):
        indices.append([int(line_splits[i][j].split(":")[0]) for j in range(1,len(line_splits[i]))])
    
    # Initializes feature matrix
    num_obs = len(indices)
    num_features = max(list((map(lambda x: int(x[len(x)-1]), indices))))+1
    num_features = 124
    
    features = np.zeros(shape=[num_obs,num_features])
    
    # Assigns '1' to appropriate indices in rows of final_array 
    #  based on values in corresponding element(list) of indices
    for i in range(len(indices)):
        features[i][indices[i]]=1
        
    features[:,0] = 1
    
    return(features, target)

X_train, y_train = load_libsvm(filepath)

X = X_train

y = y_train

X.shape

####################################################################################

def sigmoid(x):
    return(1/(1+np.exp(-x)))
#2
import numpy as np

# compute the analytic gradient
def compute_grad_obj(beta, X, y):
    """
    Return
    grad: gradient of logistic loss at beta, 1D array
    objval: objective value at beta, scalar
    """
    def sigmoid(x):
        return(1/(1+np.exp(-x)))
    
    # objval=np.sum(np.log(1+np.exp(-y*np.sum(beta.transpose()*X,axis=1))))
    
    # Overflow proof --> avoids np.log(1+np.exp(huge number))
    # objval_overflow=np.sum(np.where(-y*np.sum(beta.transpose()*X,axis=1) > 500, -y*np.sum(beta.transpose()*X,axis=1), np.log(1+np.exp(-y*np.sum(beta.transpose()*X,axis=1)))))
    objval_overflow=np.sum(np.where(-y*(X@beta.transpose()) > 500, -y*(X@beta.transpose()), np.log(1+np.exp(-y*(X@beta.transpose())))))

    # print(np.log(1+np.exp(-y*np.sum(beta.transpose()*X,axis=1))))
    grad = -(1-sigmoid(y*np.sum(beta.transpose()*X,axis=1))).reshape(-1,len(X))@(y*X.transpose()).transpose()

    return grad, objval_overflow

def sigmoid(x):
        return(1/(1+np.exp(-x)))    

# compute the analytic gradient
def compute_grad_obj_overflow(beta, X, y):
    """
    Return
    grad: gradient of logistic loss at beta, 1D array
    objval: objective value at beta, scalar
    """
    def sigmoid(x):
        return(1/(1+np.exp(-x)))
    
    y_beta_x = y*np.sum(beta.transpose()*X,axis=1)
    
    y_beta_x_capped = np.where(-y_beta_x > 700, 700, -y_beta_x)

    objval_overflow=np.sum(np.log(1+np.exp(y_beta_x_capped)))
    
    y_beta_x_floored = np.where(y_beta_x < -700, -700, y_beta_x)

    grad = -(1-sigmoid(y_beta_x_floored)).reshape(-1,len(X))@(y*X.transpose()).transpose()

    return grad, objval_overflow 


compute_grad_obj(beta,X,y)

compute_grad_obj_overflow(beta,X,y)


X = np.array([[0.75,3,2],[-2,-20000,-2],[.02,.2,2],[-.05,2,2]])
y = np.array([1,-1,1,-1])
beta = np.array([5,-3,1])



##################################################################

# backtracking line search algorithm
def backtracking_linesearch(f, grad, beta, alpha, beta_backtrack,X,y):
    """
    Parameters
    ------------
    f : objective function
    grad : gradient at beta
    beta : current parameter values
    alpha, beta : parameters of backtracking algorithm
    Return
    ------------
    eta : step size
    """
    eta = 1
    fx = f(beta,X,y)[1]
    neg_l2_norm_gradient = (grad@((-1)*grad.transpose()))[0][0]
    while f((beta - eta*grad))[0],X,y)[1] > (fx + (alpha*eta*neg_l2_norm_gradient)):
        # print(f((beta+ eta*(-1*grad))[0],X,y)[1])
        # print((f(beta,X,y)[1] + (alpha*eta*grad@((-1)*grad.transpose()))[0][0]))
        # print(eta)
        eta *= beta_backtrack

    return eta
    

beta = np.random.random(X.shape[1]) 

start = time.time()
eta = backtracking_linesearch(compute_grad_obj,compute_grad_obj(beta,X,y)[0],beta,0.4,0.8,X,y)
end = time.time()
print(end-start)


##################################################

# Now implement the gradient descent algorithm
def grad_desc(X,y,max_iter=100, step_size=0):
    """
    Parameters:
    max_iter: maximum number of iterations
    step_size: fixed step size for gradient descent. If set to zero, 
               use the backtracking line search method
    Returns:
    sol_path : a list of solutions, the kth entry corresponds to the beta 
               at iteration k
    obj_vals : a list of object values, the kth entry corresponds to the 
               objective value at iteration k               
    """
    sol_path = []
    obj_vals = []
    
    beta = np.random.random(X.shape[1]) 
    
    if step_size == 0:
        step_size = backtracking_linesearch(compute_grad_obj, compute_grad_obj(beta,X,y)[0], beta, 0.4, 0.6, X, y)

    print("Step_size: " + str(step_size))

    for i in range(max_iter):
        grad,obj = compute_grad_obj(beta,X,y)
        beta = beta - step_size*grad[0]
        sol_path.append(beta)
        obj_vals.append(obj)
        # if np.average(grad) < 0.0001:
        #     break
    return sol_path, obj_vals 


start = time.time()    
beta_path,obj_val_path = grad_desc(X,y,100,0)
end = time.time()
print(end-start)

len(beta_path)

X.shape

y.shape

# x = 6.805647338418785e-05

# z = 8.507059173023481e-05

############################################################################

# compute the conditional probability
def compute_prob(beta, x):
    """
    compute the conditional probability
    
    Parameters:
    ----------------
    beta: coefficients
    x: an example
    
    Returns:
    prob: conditional probability \sigm(beta^T x)    
    """
    return (sigmoid(np.sum(beta*x,axis=1)))

beta = np.random.random(X.shape[1])

sigmoid(np.sum(a[99]*X,axis=1))

probs = compute_prob(beta_path[99], X)

#####################################################################

filepath_test = "/Users/Miller/Documents/CSCI 3360/a9a_test.txt"

X_test, y_test = load_libsvm(filepath_test)

def predict(x, beta):
    return np.where(sigmoid(np.sum(beta*x,axis=1))>=0.5,1,-1)

preds = predict(X_test,a[-1])

preds[:10]
preds[-10:]

y_test[:10]
y_test[-10:]

#######################################################################

def compute_error(x, y, beta):    
    
    y_pred = predict(x,beta)       
    n_test = len(y_pred)
    cnt = np.count_nonzero(y_pred == y)
    
    error = 1.0 - cnt/float(n_test)

    return 100.0 * error

compute_error(X,y,a[-50])

########################################################################

import matplotlib.pyplot as plt

### Plots ###

#1.

# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.tools as tls
# tls.set_credentials_file(username='iLuvPython', api_key='w7ELl48xtF7Z7uAPIaJb')

# N = len(obj_val_path)
# plot_x = np.linspace(0, 1, N)
# plot_y = np.array(obj_val_path)

# # Create a trace
# trace = go.Scatter(
#     x = plot_x,
#     y = plot_y
# )

# data = [trace]

# py.iplot(data, filename='basic-line')

import matplotlib.pyplot as plt
plt.plot(list(range(1,len(obj_val_path)+1)),np.array(obj_val_path))
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.yticks(range(10000,210000, 20000))
plt.title('Objective Value vs. Iteration')
plt.show()

## Plot log of obj val
# plt.plot(list(range(1,len(obj_val_path)+1)),np.log(np.array(obj_val_path)))
# plt.ylabel('Objective Value')
# plt.xlabel('Iteration')
# # plt.yticks(range(int(min(obj_val_path)-1000), int(max(obj_val_path)+1000), 10000))
# plt.show()

# 2
# start = time.time()
train_error = [compute_error(X,y,beta_path[i]) for i in range(len(beta_path))]
test_error = [compute_error(X_test,y_test,beta_path[i]) for i in range(len(beta_path))]
# end = time.time()
# print(end-start)

plt.plot(list(range(1,len(train_error)+1)),np.array(train_error),'b--',list(range(1,len(test_error)+1)),np.array(test_error),'r--')
plt.ylabel('Error (%)')
plt.legend(['Train Error','Test Error'], loc="best")
plt.xlabel('Iteration')
plt.title('Error vs. Iteration')
plt.show()

# 3

num_obs = [10,100,1000,10000,30000]

# step_size = 6.0935974001049565e-05

train_indices = list(range(len(X_train)))

np.random.shuffle(train_indices)

test_indices = list(range(len(X_test)))

np.random.shuffle(test_indices)

training_errors = []
testing_errors = []
for num in num_obs:
    # if num == 10:
    #     print(X[train_indices[:num]])
    beta_path_train,obj_val_path_train = grad_desc(X_train[train_indices[:num]],y_train[train_indices[:num]],100,0)
    beta_path_test,obj_val_path_test = grad_desc(X_test[test_indices[:num]],y_test[test_indices[:num]],100,0)
    training_errors.append(compute_error(X_train,y_train,beta_path_train[-1]))
    testing_errors.append(compute_error(X_test,y_test,beta_path_test[-1]))




###############################################################################

# training_errors = []
# testing_errors = []
# for num in num_obs:
#     beta_path_train,obj_val_path_train = grad_desc(X[:num],y[:num],100,0)
#     beta_path_test,obj_val_path_test = grad_desc(X_test[:num],y_test[:num],100,0)
#     training_errors.append(compute_error(X,y,beta_path_train[-1]))
#     testing_errors.append(compute_error(X_test,y_test,beta_path_test[-1]))


N = len(num_obs)
train_errs = np.array(training_errors)
# the x locations for the groups
ind = np.array(range(N))  
# the width of the bars
width = 0.35       
fig, ax = plt.subplots()
rects1 = ax.bar(ind, train_errs, width, color='r')
test_errs = np.array(testing_errors)
rects2 = ax.bar(ind + width, test_errs, width, color='b')
ax.set_ylabel('Error')
ax.set_title('Error by Number of Observations')
ax.set_xticks(ind + width)
ax.set_xticklabels(('10', '100','1000', '10000', '30000'))
ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % round(height),
                ha='center', va='bottom')
                
autolabel(rects1)
autolabel(rects2)
plt.show()



# N = 5
# men_means = (20, 35, 30, 35, 27)
# men_std = (2, 3, 4, 1, 2)

# ind = np.arange(N)  # the x locations for the groups
# width = 0.35       # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

# women_means = (25, 32, 34, 20, 25)
# women_std = (3, 5, 2, 3, 3)
# rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# # add some text for labels, title and axes ticks
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind + width)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

# ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


# def autolabel(rects):
#     """
#     Attach a text label above each bar displaying its height
#     """
#     for rect in rects:
#         height = rect.get_height()
#         print(height)
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % height,
#                 ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

# plt.show()















