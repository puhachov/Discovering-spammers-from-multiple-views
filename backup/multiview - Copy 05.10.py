import numpy as np
import scipy
import copy
from numpy.linalg import multi_dot
import pandas as pd


class multiview():
    """
    U - base matrices
    X is passed as a list containing three multiview numpy matrices
    Y is passed as a numpy matrix
    
    regularisation_coefficients is a vector [[lambda_v], lambda_f, lambda_star_f, lambda_W]
        where:
            - [lambda_v]:    penalizes norm of the difference V*Q-V_star for each view
            - lambda_f:      penalizes sum of the squared norms of U and V
            - lambda_star_f: penalizes sum of squared norm of matrix V_star
            - lambda_W       penalizes term in SVM
    """
    
    #some const parameters for optimisation
    _MAX_ITER     = 10000          
    _TOLERANCE    = 1e-4       
    _LARGE_NUMBER = 1e100    
    
    
    
    def __init__(self, X, Y, U = None, V = None, num_components = None, view_weights = None, W = None):
        
        self.X_nv  = [np.copy(X[v]) for v in range(len(X))]
        self.n_v   = len(X)

        self.ground_truth = np.copy(Y)
        self.Y            = np.copy(Y)
        
        if view_weights is None:
            #sum(exp(-self.beta[v])) must be 1. exp(ln5) + exp(ln3) + exp(ln3) = -5 + 3 + 3 = 1. Otherwise algorithm will likely to diverge
            self.beta = np.array([-np.log(5), np.log(3), np.log(3)])
        else:
            self.beta = view_weights
            
        if num_components is None:
            self.K = 2
        else:
            self.K = num_components
            

        if U is None:
            self.U = [np.ones((self.X_nv[v].shape[0], self.K)) for v in range(self.n_v)]
        else:
            self.U = U
            
        if V is None:
            self.V = [np.ones((self.X_nv[0].shape[1], self.K)) for v in range(self.n_v)]
        else:
            self.V = V
            
        self.V_star = np.ones((self.X_nv[0].shape[1], self.K))

        self.Q = [None]*(self.n_v)


        regularisation_coefficients = None

        self.lambda_v      = None
        self.lambda_f      = None
        self.lambda_star_f = None
        self.lambda_W      = None
            
        self.alpha = None
        self.W     = np.random.random((2,self.K))
        self.eta   = None
            
    """HINGE LOSS FUNCTION ---------------------------------------------------------------------------------------------
    """
    @staticmethod
    def hinge_loss(z):
        if (z <= 0):
            return 1/2 - z
        elif (z >= 1):
            return 0
        else:
            return 1/2 * (1 - z)**2

    @staticmethod   
    def hinge_loss_derivative(z):
        if (z <= 0):
            return -1
        elif (z >= 1):
            return 0
        else:
            return z - 1
        
    """DEFINING self.CTIVE FUNCTION ----------------------------------------------------------------------------------------
    Total self.ctive Function is O = O_M + O_SVM
    """

    def _total_obj_func(self):
        
        """Calculate Q from U and V"""
        for v in range(self.n_v):
            diag_vector =  [sum(self.U[v][:,i]) for i in range(self.K)]   #i -column index 
            self.Q[v]        = np.diag(diag_vector)
        
        """Calculate multiview term O_M"""
        term_1      = [self.X_nv[v] - np.linalg.multi_dot([self.U[v],
                                            np.linalg.inv(self.Q[v]), 
                                            self.Q[v], 
                                            np.transpose(self.V[v])]) 
                       for v in range (self.n_v)]        
        term_1_norm = list(map(lambda X: scipy.linalg.norm(X, ord = 'fro')**2, term_1))
        term_2      = [self.V[v].dot(self.Q[v]) - self.V_star for v in range (self.n_v)]
        term_2_norm = list(map(lambda X: scipy.linalg.norm(X, ord = 'fro')**2, term_2))  
        term_3      = self.lambda_star_f/2 * np.linalg.norm(self.V_star, ord = 'fro')
        term_4      = [np.linalg.norm(self.U[v], ord = 'fro')**2 + np.linalg.norm(self.V[v], ord = 'fro')**2 for v in range (self.n_v)]

        O_M = 1/2 * np.sum(self.beta * term_1_norm + self.lambda_v * term_2_norm) + self.lambda_star_f * term_3 +self.lambda_f/2 * np.sum(term_4)

        """SVM Function Term"""
        l = self.Y_train.shape[0]
        S = 0
        for i in range(l):

            S += multiview.hinge_loss(self.Y_train[i,:].dot(self.W.dot(np.transpose(self.V_star[i,:]))))

        O_SVM = self.alpha * S + self.lambda_W/2 * np.linalg.norm(self.W, ord = 'fro')

        return O_M + O_SVM
    
    
    
    """OPTIMIZING W.R.T. U AND V--------------------------------------------------------------------------------------"""    
    
    def _optimize_towards_U_and_V(self): 

        iter_count     = 0        
        func_val_old   = multiview._LARGE_NUMBER
        func_val       = self._total_obj_func()
        U_old          = self.U#copy.deepcopy(self.U)
        V_old          = self.V#copy.deepcopy(self.V)

        while (iter_count < multiview._MAX_ITER)and (abs(func_val - func_val_old)/abs(func_val_old) > multiview._TOLERANCE):
            
            iter_count  += 1;
            func_val_old = func_val
            
            for v in range(self.n_v):   
                
                """UPDATE U"""
                #Resulted matrix is of size K*K, however, the size of U(v) = |U(v)|*K, where we assume |U(v)| < K. Hence we cut the matrix A to the size |U(V)|*K
                A = self.lambda_v[v] * self.beta[v] * (np.transpose(self.V[v]).dot(self.V_star))
                A = A[:self.U[v].shape[0]:,:]
                B = self.lambda_v[v] * self.beta[v] * (self.U[v].dot(np.transpose((self.V[v]**2)[:self.K,:])) + self.lambda_f * self.U[v])
                """TODO: Calculate coefficient B"""               
                numerator_U    = self.beta[v]*(self.X_nv[v].dot(self.V[v])) + A
                denominator_U  = self.beta[v] * multi_dot([self.U[v], np.transpose(self.V[v]), self.V[v]]) + B
                self.U[v]      = U_old[v] * numerator_U/denominator_U
                self.U[v]      = self.U[v]/scipy.linalg.norm(self.U[v], ord = 'fro')


                """UPDATE V"""
                numerator_V    = self.beta[v] * np.transpose(self.X_nv[v]).dot(self.U[v]) + self.lambda_v[v] * self.beta[v] * self.V_star
                denominator_V  = self.beta[v] * multi_dot([self.V[v], np.transpose(self.U[v]), self.U[v]]) + self.lambda_v[v] * self.beta[v] * self.V[v] + self.lambda_f * self.V[v]
                self.V[v]      = V_old[v] * numerator_V/denominator_V
                self.V[v]      = self.V[v]/scipy.linalg.norm(self.V[v], ord = 'fro')
                
                """UPDATE OLD U AND V """  
                V_old[v]       = self.V[v] #copy.deepcopy(self.V[v])
                U_old[v]       = self.U[v] #copy.deepcopy(self.U[v])
            #end for

            func_val  = self._total_obj_func()
            print("Iter:  {};   Old Value {}; Current Value: {}".format(iter_count, func_val_old, func_val))
        return iter_count



    def _optimize_towards_V_star_and_W(self):

        """STOCHASTIC GRADIENT DESCENT"""
        iter_count     = 0        
        func_val_old   = multiview._LARGE_NUMBER
        func_val       = self._total_obj_func()
        V_star_old     = self.V_star
        W_old          = self.W

        while (iter_count < multiview._MAX_ITER)and (abs(func_val - func_val_old)/abs(func_val_old)  > multiview._TOLERANCE):
            
            iter_count  += 1;
            func_val_old = func_val
            W_der_sum    = 0
            for i in range(self.Y_train.shape[0]):  
                """CALCULATING DERIVATIVES"""
                term_1 = 0
                for v in range(self.n_v):
                    term_1         += ( -self.lambda_v[v] * self.beta[v]) * (self.V[v][i,:].dot(self.Q[v]) - V_star_old[i,:])
                term_2              = self.alpha * multiview.hinge_loss_derivative(self.Y_train[i,:].dot(W_old).dot(np.transpose(V_star_old[i,:]))) * self.Y_train[i,:].dot(W_old)
                term_3              = self.lambda_star_f * V_star_old[i,:]
                derivative_V_star_i = term_1 + term_2 + term_3
                self.V_star[i,:]    = V_star_old[i,:] - self.learning_rate * derivative_V_star_i
                W_der_sum          += multiview.hinge_loss_derivative(self.Y_train[i,:].dot(W_old).dot(np.transpose(V_star_old[i,:]))) * (self.Y_train[i,:]).reshape((2,1)).dot((V_star_old[i,:]).reshape((1,self.K)))
            #end_for

            derivative_W = self.alpha * W_der_sum + self.lambda_W * W_old

            """UPDATING PARAMETERS"""
            self.W       =   W_old - self.learning_rate * derivative_W
            V_star_old   =   self.V_star
            W_old        =   self.W

            func_val = self._total_obj_func()
            print("Iter:  {};   Old Value {}; Current Value: {}".format(iter_count, func_val_old, func_val))
        return iter_count


    def _update_betas(self):

        """BASED ON LAGRANGE MULTIPLIER"""
        for v in range(self.n_v):
            term_1 = self.X_nv[v] - multi_dot([self.U[v],np.linalg.inv(self.Q[v]), self.Q[v], np.transpose(self.V[v])])
            term_1_norm = scipy.linalg.norm(term_1, ord = 'fro')
            term_2 = self.lambda_v[v] * scipy.linalg.norm(self.V[v].dot(self.Q[v]) - self.V_star, ord = 'fro')
            RE = term_1_norm**2 + term_2**2
            self.beta[v] = -np.log(RE/sum([term_1_norm**2 + term_2**2 for v in range(self.n_v)]))


    def solve(self, training_size, learning_rate, alpha, regularisation_coefficients = None):

        """SET UP VARIABLE PARAMETERS FOR ALGORITHM"""

        if (regularisation_coefficients is None):
            self.lambda_v      = np.array([0.5, 0.5, 0.5])
            self.lambda_f      = 0.2
            self.lambda_star_f = 0.1
            self.lambda_W      = 10
        else:
            self.lambda_v      = regularisation_coefficients[0]
            self.lambda_f      = regularisation_coefficients[1]
            self.lambda_star_f = regularisation_coefficients[2]
            self.lambda_W      = regularisation_coefficients[3]

        self.learning_rate     = learning_rate
        self.eta               = learning_rate
        self.alpha             = alpha

        """DIVIDE DATA INTO TRAINING AND TEST SET"""
        self._training_size       = int(self.ground_truth.shape[0] * training_size)

        self.Y_train              = self.ground_truth[:self._training_size,:]
        self.Y_test               = self.ground_truth[self._training_size:,:]
        self.Y_predict_train      = np.zeros(self.Y_train.shape)
        self.Y_predict_test       = np.zeros(self.Y_test.shape)


        iter_count     = 0
        iter_count_UV  = multiview._LARGE_NUMBER
        iter_count_VW  = multiview._LARGE_NUMBER      
        func_val_old   = multiview._LARGE_NUMBER
        func_val       = self._total_obj_func()

        while (iter_count_UV + iter_count_VW > 2):
            iter_count +=1
            print("Iteration {}...\n".format(iter_count))

            print("Updating U and V...\n")
            iter_count_UV = self._optimize_towards_U_and_V()
            print("DONE updating U and V...\nUpdating V_star and W...\n\n")

            iter_count_VW = self._optimize_towards_V_star_and_W()
            print("DONE updating V_star and W...\nUpdating betas...\n\n")

            self._update_betas()
            print("Done updating betas...\n\n")
            print("-------------------------------------------------------")

        print("OPTIMISATION DONE")


    def evaluate_test(self):

        for i in range(self._training_size, self.ground_truth.shape[0]):
            """PREDICTING USER'S COHORT"""
            w1_w2 = self.W.dot(np.transpose(self.V_star[i,:]))
            if (np.sum(w1_w2) < 0):
                self.Y_predict_test[self._training_size - i,:] = np.array([-1., 0.])
            else:
                self.Y_predict_test[self._training_size - i,:] = np.array([0., 1.])

        """CONFUSION MATRIX TP|FP
                            TN|FN
        """ 
        confusion_matrix = np.zeros((2,2))
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(self.Y_predict_test.shape[0]):
            if (np.array_equal(self.Y_test[i,:], self.Y_predict_test[i,:]))and(np.array_equal(self.Y_predict_test[i,:], np.array([-1., 0.]))):
                TP += 1
            if (np.array_equal(self.Y_test[i,:], self.Y_predict_test[i,:]))and(np.array_equal(self.Y_predict_test[i,:], np.array([0., 1.]))):
                TN += 1
            if (np.array_equal(self.Y_test[i,:], np.array([-1., 0.])))and(np.array_equal(self.Y_predict_test[i,:], np.array([0., 1.]))):
                FP += 1
            if (np.array_equal(self.Y_test[i,:], np.array([0., 1.])))and(np.array_equal(self.Y_predict_test[i,:], np.array([-1., 0.]))):
                FN += 1

        confusion_matrix = pd.DataFrame(data = {'Actual_Spammer': [TP, FN], 'Actual_Legitimate': [FP, TN]}, index = ['Predicted_Spammer ','Predicted_Legitimate'])
        precision        = TP/(TP+FP)
        recall           = TP/(TP+FN)
        F1_score         = 2*TP/(2*TP + FP + FN)

        return confusion_matrix, precision, recall, F1_score



    def evaluate_train(self):

        for i in range(self.Y_train.shape[0]):
            """PREDICTING USER'S COHORT"""
            w1_w2 = self.W.dot(np.transpose(self.V_star[i,:]))
            if (np.sum(w1_w2) < 0):
                self.Y_predict_train[i,:] = np.array([-1., 0.])
            else:
                self.Y_predict_train[i,:] = np.array([0., 1.])

        """CONFUSION MATRIX TP|FP
                            TN|FN
        """ 
        confusion_matrix = np.zeros((2,2))
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(self.Y_predict_train.shape[0]):
            if (np.array_equal(self.Y_train[i,:], self.Y_predict_train[i,:]))and(np.array_equal(self.Y_predict_train[i,:], np.array([-1., 0.]))):
                TP += 1
            if (np.array_equal(self.Y_train[i,:], self.Y_predict_train[i,:]))and(np.array_equal(self.Y_predict_train[i,:], np.array([0., 1.]))):
                TN += 1
            if (np.array_equal(self.Y_train[i,:], np.array([-1., 0.])))and(np.array_equal(self.Y_predict_train[i,:], np.array([0., 1.]))):
                FP += 1
            if (np.array_equal(self.Y_train[i,:], np.array([0., 1.])))and(np.array_equal(self.Y_predict_train[i,:], np.array([-1., 0.]))):
                FN += 1

        confusion_matrix = pd.DataFrame(data = {'Actual_Spammer': [TP, FN], 'Actual_Legitimate': [FP, TN]}, index = ['Predicted_Spammer ','Predicted_Legitimate'])
        precision        = TP/(TP+FP)
        recall           = TP/(TP+FN)
        F1_score         = 2*TP/(2*TP + FP + FN)

        return confusion_matrix, precision, recall, F1_score 









