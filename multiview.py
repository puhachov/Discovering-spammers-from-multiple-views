import numpy as np
import scipy
import copy
from numpy.linalg import multi_dot, inv, norm
import pandas as pd
from numpy.random import uniform


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
        
        if view_weights is None:
            #sum(exp(-self.beta[v])) must be 1. exp(ln5) + exp(ln3) + exp(ln3) = -5 + 3 + 3 = 1. Otherwise algorithm will likely to diverge
            self.beta = np.array([np.log(3)] * self.n_v)
        else:
            self.beta = view_weights
            
        if num_components is None:
            self.K = 2
        else:
            self.K = num_components
            
        np.random.seed(int(uniform(1,100)))
        if U is None:
            self.U = [np.random.random((self.X_nv[v].shape[0], self.K)) for v in range(self.n_v)]
        else:
            self.U = U
            
        if V is None:
            self.V = [np.random.random((self.X_nv[0].shape[1], self.K)) for v in range(self.n_v)]
        else:
            self.V = V
            
        self.V_star = np.random.random((self.X_nv[0].shape[1], self.K))
        self.Q = [np.diag(np.sum(self.U[v], axis = 0)) for v in range(self.n_v)]

        regularisation_coefficients = None

        self.lambda_v      = None
        self.lambda_f      = None
        self.lambda_star_f = None
        self.lambda_W      = None
            
        self.alpha         = None
        self.W             = np.random.random((2,self.K))
        self.learning_rate = None
        self.confusion_matrix_te = None
        self.confusion_matrix_tr = None
            
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

       
    """DEFINING OBJECTIVE FUNCTION ----------------------------------------------------------------------------------------
    Total objective Function is O = O_M + O_SVM
    """

    def _total_obj_func(self):

        """Calculate multiview term O_M"""
        term_1      = [self.X_nv[v] - multi_dot([self.U[v],
                                            inv(self.Q[v]), 
                                            self.Q[v], 
                                            self.V[v].T]) 
                       for v in range (self.n_v)]        
        term_1_norm_sq = [norm(X, ord = 'fro')**2   for X in term_1]
        term_2         = [self.V[v].dot(self.Q[v]) - self.V_star for v in range (self.n_v)]
        term_2_norm_sq = [norm(X, ord = 'fro')      for X in term_2]
        term_3_norm_sq = norm(self.V_star, ord = 'fro')**2
        term_4         = [norm(self.U[v], ord = 'fro')**2 + norm(self.V[v], ord = 'fro')**2 for v in range (self.n_v)]

        #O_M = 1/2 * np.sum(np.multiply(self.beta,  np.sum(term_1_norm_sq, np.multiply(self.lambda_v, term_2_norm_sq)))) + self.lambda_star_f/2 * term_3_norm_sq + self.lambda_f/2 * np.sum(term_4)
        O_M  = 1/2 * np.sum(np.multiply(self.beta, np.add(term_1_norm_sq, np.multiply(self.lambda_v, term_2_norm_sq)))) + self.lambda_star_f/2 * term_3_norm_sq + self.lambda_f/2 * np.sum(term_4)
        
        """SVM Function Term"""
        S = 0
        for i in range(self.Y_train.shape[0]):
            S += multiview.hinge_loss(self.Y_train[i,:].dot(self.W.dot(self.V_star[i,:].T)))

        O_SVM = self.alpha * S + self.lambda_W/2 * norm(self.W, ord = 'fro')**2

        return O_M + O_SVM
    
    
    
    """OPTIMIZING W.R.T. U AND V--------------------------------------------------------------------------------------"""    
    
    def _optimize_towards_U_and_V(self): 

        views_dict = {0:"Content View", 1: "URL View", 2: "Hashtag View"}
        for v in range(self.n_v):   

            print("Partial Optimisation w.r.t. {}".format(views_dict[v]))
            iter_count     = 0        
            func_val_old   = multiview._LARGE_NUMBER
            func_val       = self._total_obj_func()
            #U_old          = copy.deepcopy(self.U)
            #V_old          = copy.deepcopy(self.V)
            
            while (iter_count < multiview._MAX_ITER)and (abs(func_val - func_val_old)/abs(func_val_old) > multiview._TOLERANCE):

                """UPDATE U"""
                A = self.lambda_v[v] * self.beta[v] * np.tile(np.sum(np.multiply(self.V[v], self.V_star), axis = 0), (self.U[v].shape[0], 1))
                B = self.lambda_v[v] * self.beta[v] * np.tile(np.multiply(np.sum(self.U[v], axis = 0), np.sum(self.V[v]**2, axis = 0)), (self.U[v].shape[0], 1)) + self.lambda_f * self.U[v]
                

                numerator_U    = self.beta[v] * (self.X_nv[v].dot(self.V[v])) + A
                denominator_U  = self.beta[v] * multi_dot([self.U[v], self.V[v].T, self.V[v]]) + B
                self.U[v]      = self.U[v]    * (numerator_U/denominator_U)

                """UPDATE Q"""
                self.Q         = [np.diag(np.sum(self.U[v], axis = 0)) for v in range(self.n_v)]    

                """NORMALIZE U AND V"""

                self.U[v]      = self.U[v].dot(inv(self.Q[v]))
                self.V[v]      = self.V[v].dot(self.Q[v])

                #print(inv(self.Q[v]), self.Q[v])
                #self.U[v]      = self.U[v].dot(inv(self.Q[v]))
                #self.V[v]      = self.V[v].dot(self.Q[v])


                """UPDATE V"""
                numerator_V    = self.beta[v] * (self.X_nv[v].T).dot(self.U[v]) + self.lambda_v[v] * self.beta[v] * self.V_star
                denominator_V  = self.beta[v] * multi_dot([self.V[v], self.U[v].T, self.U[v]]) + self.lambda_v[v] * self.beta[v] * self.V[v] + self.lambda_f * self.V[v]
                self.V[v]      = self.V[v]    * (numerator_V/denominator_V)

                #print("U:\n {}\n\n V:\n {}\n\n numerator_V:\n {}\n\n  V_star:\n {}\n\n \n\n".format(self.U[v][:10,:10], self.V[v][:10,:10], numerator_V[:10,:10], self.V_star[:10,:10]))

                #U_old[v] = self.U[v]
                #V_old[v] = self.V[v]
                iter_count  += 1;
                func_val_old = func_val
                func_val  = self._total_obj_func()
                
                print("Iter:  {};   Old Value {}; Current Value: {}".format(iter_count, func_val_old, func_val))
            #end while
            print("------------------------------------------------------------------------------------")
        #return iter_count



    def _optimize_towards_V_star_and_W(self):

        """STOCHASTIC GRADIENT DESCENT"""
        iter_count     = 0        
        func_val_old = delta_V_star  = delta_W  = multiview._LARGE_NUMBER
        func_val       = self._total_obj_func()
        #W_old        = np.copy(self.W)
        #V_star_old   = np.copy(self.V_star)

        while (iter_count < multiview._MAX_ITER)and (abs(func_val - func_val_old)/abs(func_val_old)  > multiview._TOLERANCE):
        #while (iter_count < multiview._MAX_ITER)and (delta_V_star  > 0.1)or( delta_W > 0.01)and(abs(func_val - func_val_old)/abs(func_val_old)  > multiview._TOLERANCE):
            

            W_der_sum    = 0
            W_old        = copy.deepcopy(self.W)
            V_star_old   = copy.deepcopy(self.V_star)
            #-------------------------------------------------------CALCULATING NEW VALUES OF DERIVATIVES----------------------------------------
            for i in range(self.Y_train.shape[0]):  
                """CALCULATING DERIVATIVES"""
                term_1              = np.sum([-self.lambda_v[v] * self.beta[v] * (self.V[v][i,:].dot(self.Q[v]) - V_star_old[i,:]) for v in range(self.n_v)])
                term_2              = self.alpha * multiview.hinge_loss_derivative(self.Y_train[i,:].dot(W_old).dot(np.transpose(V_star_old[i,:]))) * self.Y_train[i,:].dot(W_old)
                term_3              = self.lambda_star_f * V_star_old[i,:]
                derivative_V_star_i = term_1 + term_2 + term_3
                self.V_star[i,:]    = self.V_star[i,:] - self.learning_rate * derivative_V_star_i
                self.V_star[i,:][self.V_star[i,:] < 0 ] = 0

                #we use the new value of V_star to calculate a derivative (paper states update is SEQUENTIAL)
                W_der_sum          += multiview.hinge_loss_derivative(self.Y_train[i,:].dot(W_old).dot(np.transpose(V_star_old[i,:]))) * (self.Y_train[i,:]).reshape((2,1)).dot((V_star_old[i,:]).reshape((1,self.K)))
            #end_for

            derivative_W = self.alpha * W_der_sum + self.lambda_W * self.W
            #-------------------------------------------------------UPDATING OLD VALUES AND CALCULATING DIFFERENCE-------------------------

            """UPDATING PARAMETERS"""
            self.W       = self.W - self.learning_rate * derivative_W

            
            delta_V_star = norm(V_star_old - self.V_star, ord = 'fro')
            delta_W      = norm(W_old - self.W, ord = 'fro')
            #print(delta_V_star, delta_W)

            iter_count  += 1;
            func_val_old = func_val
            func_val = self._total_obj_func()
            print("Iter:  {};   Old Value {}; Current Value: {}".format(iter_count, func_val_old, func_val))
            #-----------------------------------------------------------GD ITERATION DONE------------------------------
        #return iter_count


    def _update_betas(self):

        """BASED ON LAGRANGE MULTIPLIER"""
        for v in range(self.n_v):
            term_1       = self.X_nv[v] - multi_dot([self.U[v],inv(self.Q[v]), self.Q[v], np.transpose(self.V[v])])
            term_1_norm  = norm(term_1, ord = 'fro')
            term_2       = norm(self.V[v].dot(self.Q[v]) - self.V_star, ord = 'fro')
            RE           = term_1_norm**2 + self.lambda_v[v] * term_2**2
            sum_         = sum([RE for v in range(self.n_v)])
            self.beta[v] = -np.log(RE/sum_)


    def solve(self, training_size, learning_rate, alpha, regularisation_coefficients = None):

        """SET UP VARIABLE PARAMETERS FOR ALGORITHM"""

        if (regularisation_coefficients is None):

            self.lambda_v      = np.array([0.1] * self.n_v)

            #self.lambda_v      = np.array([3,2,2])
            self.lambda_v      = np.array([0.1, 0.1, 0.1])

            self.lambda_f      = 0.1
            self.lambda_star_f = 0.5
            self.lambda_W      = 0.4
        else:
            self.lambda_v      = regularisation_coefficients[0]
            self.lambda_f      = regularisation_coefficients[1]
            self.lambda_star_f = regularisation_coefficients[2]
            self.lambda_W      = regularisation_coefficients[3]

        self.learning_rate     = learning_rate
        self.alpha             = alpha

        """DIVIDE DATA INTO TRAINING AND TEST SET"""    
        self._training_size       = int(self.ground_truth.shape[0] * training_size)

        self.Y_train              = self.ground_truth[:self._training_size,:]
        self.Y_test               = self.ground_truth[self._training_size:,:]
        self.Y_p_train            = np.zeros(self.Y_train.shape)
        self.Y_p_test             = np.zeros(self.Y_test.shape)         
        #while (iter_count < multiview._MAX_ITER)and (abs(func_val - func_val_old)/abs(func_val_old)  > multiview._TOLERANCE):
        iter_count     = 0    
        func_val_old   = multiview._LARGE_NUMBER
        func_val       = self._total_obj_func()

        while (iter_count < multiview._MAX_ITER)and (abs(func_val - func_val_old)/abs(func_val_old)  > multiview._TOLERANCE):
            iter_count +=1
            func_val_old = func_val

            print("Updating U and V...\n")
            self._optimize_towards_U_and_V()
            print("DONE updating U and V...\nUpdating V_star and W...\n\n")

            self._optimize_towards_V_star_and_W()
            print("DONE updating V_star and W...\nUpdating betas...\n\n")

            self._update_betas()
            print("Done updating betas...\nCalculating Global objective function value...\n\n")
            func_val = self._total_obj_func()
            print("Iter:  {};   Old Value {}; Current Value: {}".format(iter_count, func_val_old, func_val))
            print("-------------------------------------------------------")
        print("OPTIMISATION DONE")


    def evaluate_test(self):

        for i in range(self._training_size, self.ground_truth.shape[0]):
            """PREDICTING USER'S COHORT"""
            w1_w2 = self.W.dot(np.transpose(self.V_star[i,:]))
            if (np.sum(w1_w2) < 0):
                self.Y_p_test[self._training_size - i,:] = np.array([-1., 0.])
            else:
                self.Y_p_test[self._training_size - i,:] = np.array([0., 1.])

        """CONFUSION MATRIX TP|FP
                            TN|FN
        """ 
        confusion_matrix = np.zeros((2,2))
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(self.Y_p_test.shape[0]):
            if (np.array_equal(self.Y_test[i,:], self.Y_p_test[i,:]))and(np.array_equal(self.Y_p_test[i,:], np.array([-1., 0.]))):
                TP += 1
            if (np.array_equal(self.Y_test[i,:], self.Y_p_test[i,:]))and(np.array_equal(self.Y_p_test[i,:], np.array([0., 1.]))):
                TN += 1
            if (np.array_equal(self.Y_test[i,:], np.array([-1., 0.])))and(np.array_equal(self.Y_p_test[i,:], np.array([0., 1.]))):
                FP += 1
            if (np.array_equal(self.Y_test[i,:], np.array([0., 1.])))and(np.array_equal(self.Y_p_test[i,:], np.array([-1., 0.]))):
                FN += 1


        confusion_matrix_te_ = pd.DataFrame(data = {'Actual_Spammer': [TP, FN], 'Actual_Legitimate': [FP, TN]}, index = ['Predicted_Spammer ','Predicted_Legitimate'])
        precision           = confusion_matrix_te_[0,0]/(confusion_matrix_te_[0,0] + confusion_matrix_te_[0,1])
        recall              = confusion_matrix_te_[0,0]/(confusion_matrix_te_[0,0] + confusion_matrix_te_[1,0])
        F1_score            = 2*precision*recall/(precision + recall)

        return confusion_matrix_te_, precision, recall, F1_score



    def evaluate_train(self):

        for i in range(self.Y_train.shape[0]):
            """PREDICTING USER'S COHORT"""
            w1_w2 = self.W.dot(np.transpose(self.V_star[i,:]))
            if (np.sum(w1_w2) < 0):
                self.Y_p_train[i,:] = np.array([-1., 0.])
            else:
                self.Y_p_train[i,:] = np.array([0., 1.])

        """CONFUSION MATRIX TP|FP
                            TN|FN
        """ 
        confusion_matrix = np.zeros((2,2))
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(self.Y_p_train.shape[0]):
            if (np.array_equal(self.Y_train[i,:], self.Y_p_train[i,:]))and(np.array_equal(self.Y_p_train[i,:], np.array([-1., 0.]))):
                TP += 1
            if (np.array_equal(self.Y_train[i,:], self.Y_p_train[i,:]))and(np.array_equal(self.Y_p_train[i,:], np.array([0., 1.]))):
                TN += 1
            if (np.array_equal(self.Y_train[i,:], np.array([-1., 0.])))and(np.array_equal(self.Y_p_train[i,:], np.array([0., 1.]))):
                FP += 1
            if (np.array_equal(self.Y_train[i,:], np.array([0., 1.])))and(np.array_equal(self.Y_p_train[i,:], np.array([-1., 0.]))):
                FN += 1


        confusion_matrix_tr_ = pd.DataFrame(data = {'Actual_Spammer': [TP, FN], 'Actual_Legitimate': [FP, TN]}, index = ['Predicted_Spammer ','Predicted_Legitimate'])
        precision           = confusion_matrix_tr_[0,0]/(confusion_matrix_tr_[0,0] + confusion_matrix_tr_[0,1])
        recall              = confusion_matrix_tr_[0,0]/(confusion_matrix_tr_[0,0] + confusion_matrix_tr_[1,0])
        F1_score            = 2*precision*recall/(precision + recall)

        return confusion_matrix_tr_, precision, recall, F1_score 


    def evaluate_train_sklearn(self):


        for i in range(self.Y_train.shape[0]):
            """PREDICTING USER'S COHORT"""
            w1_w2 = self.W.dot(np.transpose(self.V_star[i,:]))
            if (np.sum(w1_w2) < 0):
                self.Y_p_train[i,:] = np.array([-1., 0.])
            else:
                self.Y_p_train[i,:] = np.array([0., 1.])

        Y_p_train_cust      = np.sum(self.Y_p_train, axis = 1)
        Y_train_cust        = np.sum(self.Y_train, axis = 1)

        confusion_matrix_   = confusion_matrix(Y_train_cust, Y_p_train_cust)
        precision           = confusion_matrix_[0,0]/(confusion_matrix_[0,0] + confusion_matrix_[0,1])
        recall              = confusion_matrix_[0,0]/(confusion_matrix_[0,0] + confusion_matrix_[1,0])
        F1_score            = 2*precision*recall/(precision + recall)
        confusion_matrix_df = pd.DataFrame(data    = confusion_matrix_,
                                           columns = ['Actual_Spammer', 'Actual_Legitimate'],
                                           index   = ['Predicted_Spammer ','Predicted_Legitimate'])
        print("Precision: {}\n".format(precision))
        print("Recall: {}\n".format(recall))
        print("F1-score: {}\n".format(F1_score))
        print("Confusion Matrix: {}\n".format(confusion_matrix_))
        return precision, recall, F1_score, confusion_matrix_










