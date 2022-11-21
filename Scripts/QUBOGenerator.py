#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB
import math
from math import prod
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model
from qubovert.problems import BILP
from dimod import BinaryQuadraticModel


# In[2]:


## Takes the logarithms of the coefficient values and rounds the values to the nearest decimal place in accordance to the
## num_decimal_pos parameter
def get_log_values(coeff, num_decimal_pos):
    log_coeff = np.around(np.log10(coeff), num_decimal_pos)
    return log_coeff.tolist()

## Computes and returns the logarithmic maximum intermediate cardinality for the join corresponding to the given join_index based
## on the given sorted_card list
def get_maximum_log_intermediate_outer_operand_cardinality(join_index, sorted_card):
    return sum(sorted_card[i] for i in range(join_index+1))

## Computes and returns the (non-logarithmic) maximum intermediate cardinality for the join corresponding to the given join_index based
## on the given sorted_card list
def get_maximum_intermediate_outer_operand_cardinality(join_index, sorted_card):
    return prod(sorted_card[i] for i in range(join_index+1))

## Returns a dictionary containing for each threshold value and join the large constant needed to make constraint 7 satisfiable in any case.
## In order to make the constraint for the threshold value i and join j satisfiable, C needs to have the following lower bound:
## C >= mlc_j - log_thres_i, where mlc_j is the maximum intermediate logarithmic cardinality for join j
def get_large_threshold_coefficients(num_thres, num_joins, log_thres, card):
    # Sort the cardinalities in descending order
    sorted_card = sorted(card, reverse=True)
    coeff_dict = {}
    for i in range(num_thres):
        for j in range(num_joins):
            coeff_dict[(i, j)] = get_maximum_log_intermediate_outer_operand_cardinality(j, sorted_card) - log_thres[i]
    return coeff_dict
    
## Returns a dictionary containing for each threshold value i and join j the number of slack variables needed for
## constraint 7 (which equals log_2(log_thres[i] + large_threshold_coefficients[(i, j)]), where log_thres contains
## the logarithmic threshold values and large_threshold_coefficients the large constants needed to make constraint 7 satisfiable)
def get_cnstr7_slack_variable_numbers(num_thres, num_joins, log_thres, slack_upper_bound_dict, precision_weight):
    coeff_dict = {}
    for i in range(num_thres):
        for j in range(num_joins):
            upper_bound = slack_upper_bound_dict[(i, j)]
            # Check if the constraint exists for threshold value i and join j
            if (upper_bound != 0):
                exp = int(math.floor(np.log2(upper_bound/precision_weight))) + 1
                coeff_dict[(i, j)] = exp
    return coeff_dict

## Returns a dictionary containing all exponentials of 2 up until the maximum exponent within the given dictionary
def get_cnstr7_slack_exponential_dictionary(num_cnstr7_slack_variables):
    exp_dict = {}
    if len(num_cnstr7_slack_variables.values()) == 0:
        return exp_dict
    exp_range = range(max(num_cnstr7_slack_variables.values()))
    for i in exp_range:
        exp_dict[i] = pow(2, i)
    return exp_dict

def calculate_invalidation_hamiltonian_weight(delta_thres, cto_variable_dict, precision_weight, weight_b):
    epsilon = 0.25
    accumulated_cost = 0
    for (i, j) in cto_variable_dict.keys():
        accumulated_cost = accumulated_cost + delta_thres[i]
    return (accumulated_cost/pow(precision_weight, 2))*weight_b + epsilon

def is_variable_pruned(thres_index, join_index, sorted_card, thres, enable_pruning):
    if not enable_pruning:
        return False
    maxcard = get_maximum_intermediate_outer_operand_cardinality(join_index, sorted_card)
    if maxcard <= thres[thres_index]:
        return True
    return False

## Returns a dictionary which contains for a tuple (r, j) where r denotes the r-the threshold value and j denotes
## the j-th join the upper bound C for the continuous slack variable of a type 7 constraint.
## If it is determined that the variable cto_rj and thus the constraint is unnecessary, the value for (r, j) will be 0 instead.
def calculate_slack_upper_bound_dict(thres, card, log_thres, log_card, enable_pruning):
    upper_bound_dict = {}
    sorted_card = sorted(card, reverse=True)
    sorted_log_card = sorted(log_card, reverse=True)
    for i in range(len(log_thres)):
        for j in range(len(log_card)-1):
            maxlogcard = get_maximum_log_intermediate_outer_operand_cardinality(j, sorted_log_card)
            if is_variable_pruned(i, j, sorted_card, thres, enable_pruning):
                upper_bound_dict[(i, j)] = 0
            else:
                upper_bound_dict[(i, j)] = maxlogcard
    return upper_bound_dict
    
## Returns a dictionary containing a variable cto_rj for a threshold value r and a join j.
## However, the dictionary only includes entries for non-pruned r and j in accordance to the given slack_upper_bound_dict
def get_cto_variable_dict(model, slack_upper_bound_dict): 
    cto_variable_dict = {}
    for (i, j) in slack_upper_bound_dict.keys():
        ## As we are only interested in the intermediate cardinalities, we do not need cto variables for the 
        ## first join (as the outer operand is merely one of the input cardinalities)
        if (j != 0) and (slack_upper_bound_dict[(i, j)] != 0):
            cto_variable_dict[(i, j)] = model.addVar(vtype=GRB.BINARY, name="cto" + str(i) + str(j))
    return cto_variable_dict

## Returns a list of delta threshold values such that no threshold value is added twice in the objective function
def get_delta_thres(thres):
    delta_thres = np.diff(thres)
    delta_thres = np.insert(delta_thres, 0, thres[0])
    return delta_thres


# In[3]:


def generate_BILP_model(card, thres, num_decimal_pos, pred, pred_sel, enable_pruning):
    ##----------------------------------------------------------------------------------------------------
    ## Generating the necessary helper data structures

    num_tables = len(card)
    num_joins = len(card)-1
    num_thres = len(thres)
    num_preds = len(pred_sel)
    delta_thres = get_delta_thres(thres)
    precision_weight = pow(0.1, num_decimal_pos)

    log_cards = get_log_values(card, num_decimal_pos)
    log_pred_sel = get_log_values(pred_sel, num_decimal_pos)
    log_thres = get_log_values(thres, num_decimal_pos)
    
    slack_upper_bound_dict = calculate_slack_upper_bound_dict(thres, card, log_thres, log_cards, enable_pruning)

    large_threshold_coefficients = get_large_threshold_coefficients(num_thres, num_joins, log_thres, log_cards)
    num_cnstr7_slack_variables = get_cnstr7_slack_variable_numbers(num_thres, num_joins, log_thres, slack_upper_bound_dict, precision_weight)
    cnstr7_exp_dictionary = get_cnstr7_slack_exponential_dictionary(num_cnstr7_slack_variables)


    ##----------------------------------------------------------------------------------------------------
    ## Formulating the Binary Integer Linear Programming problem using Gurobi
    
    m = gp.Model("bip")

    ## Adding the binary variables to the model

    tio = m.addVars(num_tables, num_joins, vtype=GRB.BINARY, name="tio") # Adding the tio binary variables to the model
    tii = m.addVars(num_tables, num_joins, vtype=GRB.BINARY, name="tii") # Adding the tio binary variables to the model
    pao = m.addVars(num_preds, num_joins-1, vtype=GRB.BINARY, name="pao") # Adding the pao binary variables to the model
    cto_variable_dict = get_cto_variable_dict(m, slack_upper_bound_dict)
    
    ## Setting the objective: Minimizing the accumulated approximate cardinality for the outer join operands

    m.setObjective(gp.quicksum(delta_thres[i]*cto_variable_dict[(i, j)] 
                    for (i,j) in cto_variable_dict.keys()), GRB.MINIMIZE)

    ## Adding the constraints
    ## Constr. 1 and 2: Select one table for outer operand of first join/for all inner operands

    m.addConstrs(tii.sum('*', i) == 1 for i in range(num_joins))
    m.addConstr(tio.sum('*', 0) == 1)

    ## Constr. 3: The tables in the join operands must not overlap for the same join
    ## Note: this constraint is mostly already encoded in constr. 4 -> merely for the final join, it is still required

    ## Inequality constraint: Introducing num_joins * num_tables slack variables for transformation into equality constraint

    sl_cnstr2 = m.addVars(num_tables, vtype=GRB.BINARY, name="sl_cnstr2")
    #sl_cnstr2 = m.addVars(num_tables, num_joins, vtype=GRB.BINARY, name="sl_cnstr2")

    #m.addConstrs(tio[i, j] + tii[i, j] + sl_cnstr2[i, j] == 1 for i in range(num_tables) for j in range(num_joins))
    m.addConstrs(tio[i, num_joins-1] + tii[i, num_joins-1] + sl_cnstr2[i] == 1 for i in range(num_tables))


    ## Constr. 4: Results of prior joins are outer operands for the next join

    m.addConstrs(tio[i, j] == tii[i, j-1] + tio[i, j-1] for i in range(num_tables) for j in range(1, num_joins))

    ## Constr. 5 and 6: Predicates are only applicable if both referenced tables are in the outer operand

    ## Inequality constraint: Introducing 2 * num_preds * num_joins slack variables for transformation into equality constraint

    sl_cnstr4_1 = m.addVars(num_preds, num_joins-1, vtype=GRB.BINARY, name="sl_cnstr4_1")

    m.addConstrs(pao[i, j-1] + sl_cnstr4_1[i, j-1] == tio[pred[i][0], j] for i in range(num_preds) for j in range(1, num_joins))

    sl_cnstr4_2 = m.addVars(num_preds, num_joins-1, vtype=GRB.BINARY, name="sl_cnstr4_2")

    m.addConstrs(pao[i, j-1] + sl_cnstr4_2[i, j-1] == tio[pred[i][1], j] for i in range(num_preds) for j in range(1, num_joins))

    ## Constr. 7: Activates threshold variable if cardinality reaches threshold

    ## Inequality constraint: Introducing num_thres * num_joins sets of auxiliary variables of arbitrary size (as specified in the num_cnstr7_slack_variables dictionary)
    ## Combined with the cnstr7_precision_weight containing a weight factor for each set of aux. variables, this allows for the approximatio of continuous variables via
    ## binary variables in order to transform the inequality constraint into an equality constraint. The approximation precision depends on the specified precision weight

    
    ## For each non-pruned cto variable, a type 7 constraint is added to the model
    for (i, j) in cto_variable_dict.keys():
        ## Before that, the binary variables for approximating the continuous variables need to be added as well
        sl_cnstr7 = m.addVars(num_cnstr7_slack_variables[(i, j)], vtype=GRB.BINARY, name="sl_cnstr7_" + str(i) + str(j))

        cto = cto_variable_dict[(i, j)]

        m.addConstr(gp.quicksum(log_cards[l]*tio[l, j] for l in range(num_tables)) 
                 + gp.quicksum(log_pred_sel[l]*pao[l, j-1] for l in range(num_preds))
                 - cto *large_threshold_coefficients[(i, j)]
                 + precision_weight * gp.quicksum(sl_cnstr7[l]*cnstr7_exp_dictionary[l] for l in range(num_cnstr7_slack_variables[(i, j)]))
                 == log_thres[i])

        
    ## After adding all the constraints, the model needs to be updated. Before this, the changes to the constraints are only buffered
    m.update()
    
    ## Finally, we calculate the penalty weights required to transform the BILP model into QUBO
    
    weight_b = 1
    weight_a = calculate_invalidation_hamiltonian_weight(delta_thres, cto_variable_dict, precision_weight, weight_b)
    
    return m, weight_a, weight_b


# In[4]:


def generate_QUBO_for_IBMQ(card, thres, num_decimal_pos, pred, pred_sel, enable_pruning=True):
    
    ## Generate the MILP/BILP model
    
    bilp, weight_a, weight_b = generate_BILP_model(card, thres, num_decimal_pos, pred, pred_sel, enable_pruning)
    
    ## Extract the coefficent matrix (A) from model                

    A = bilp.getA().toarray()
    np.set_printoptions(threshold=np.inf)
    
    ## Extract the b vector (the right hand side for the equality constraints) from the model

    constrs = bilp.getConstrs()

    mapper = lambda c: c.rhs
    vfunc = np.vectorize(mapper)
    right_hand_side = vfunc(constrs)
    #print("Number of constraints:")
    #print(len(right_hand_side))

    ## Extract the c vector (the coefficients for the objective function) from the model

    dvars = bilp.getVars()
    c = bilp.getAttr('Obj', dvars)
    
    ##----------------------------------------------------------------------------------------------------
    ## Transforming the gurobi model into a quadratic program
    
    num_qubits = len(bilp.getVars())
    num_constr = len(right_hand_side)

    # Defining the variables and hamiltonians

    model = Model('docplex_model')

    v = model.binary_var_list(num_qubits)

    H_A = weight_a* model.sum((right_hand_side[j] - model.sum(A[j][i] * v[i] for i in range(num_qubits)))**2 for j in range(num_constr))

    H_B = model.sum(c[i] * v[i] * weight_b for i in range(num_qubits))

    H = H_A + H_B

    model.minimize(H)

    qubo = from_docplex_mp(model)
    
    return qubo, weight_a

def generate_QUBO_for_DWave(card, thres, num_decimal_pos, pred, pred_sel, enable_pruning=True):
    
    bilp, weight_a, weight_b = generate_BILP_model(card, thres, num_decimal_pos, pred, pred_sel, enable_pruning)
    
    ## Extract the coefficent matrix (A) from model                

    A = bilp.getA().toarray()
    np.set_printoptions(threshold=np.inf)
    
    ## Extract the b vector (the right hand side for the equality constraints) from the model

    constrs = bilp.getConstrs()

    mapper = lambda c: c.rhs
    vfunc = np.vectorize(mapper)
    right_hand_side = vfunc(constrs)
    #print("Number of constraints:")
    #print(len(right_hand_side))

    ## Extract the c vector (the coefficients for the objective function) from the model

    dvars = bilp.getVars()
    c = bilp.getAttr('Obj', dvars)
    
    #print("Number of variables: " + str(len(dvars)))
    
    ## Solve with gurobipy
    
    #m.optimize()
    
    #print('Obj: %g' % m.objVal)
    
    #for v in m.getVars():
        #print('%s %g' % (v.varName, v.x))

    ##----------------------------------------------------------------------------------------------------
    ## Transforming the gurobi model into into BQM/QUBO
    
    qubo = BILP(c, A, right_hand_side).to_qubo(A=weight_a, B=weight_b)
    
    bqm = BinaryQuadraticModel.from_qubo(Q=qubo.Q, offset=qubo.offset)

    return bqm, weight_a

