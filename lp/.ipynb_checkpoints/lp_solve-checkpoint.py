import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#solves LP problem for ONLY ONE optimization window e.g for one DAY
#henergy is the array with the energy harvest for one day (i.e has the length of one time optimization window)

# HMAX = 1000 #Maximum harvested energy


# DMIN = 1 #20% duty cycle = 100 mWhr
# DMAX = 10 #100% duty cycle = 500 mWhr
# DSCALE = 50 #scale to convert action value to actual power consumption
# NMAX = DMAX * DSCALE #max energy consumption
# BMIN = 0.0
# BMAX = 9250.0
# BOPT = 0.6 * BMAX
# BINIT = 0.6 * BMAX
    
def solve(henergy):
    SLOTS = len(henergy)

    optimization_flag = None #0 = lowest duty cycle; 1 = LP solution; 2 = highest duty cycle;

    #the epochs in a window of SLOTS numbers of slots
    epoch = ['epoch_%d' %i for i in range(1, SLOTS + 1, 1)]

    #Create dictionary of harvested energy
    henergy_dict = dict(zip(epoch, henergy))


    if sum(henergy) < DMIN*DSCALE*SLOTS:
        #print("Always on lowest duty cycle")
        optimization_flag = 0
    elif sum(henergy) > DMAX*DSCALE*SLOTS:
        #print("Always on highest duty cycle")
        optimization_flag = 2
    else:
        #Define the LP problem as "ENO" as type Minimization
        model = pulp.LpProblem('ENO', pulp.LpMinimize)


        # create a dictionary of pulp LpVariables with keys corresponding to values in the list epoch
        action_dict = pulp.LpVariable.dicts('action', epoch , 
                                           lowBound=DMIN, upBound=DMAX, 
                                           cat=pulp.LpInteger)


        total_consumed_energy = pulp.lpSum([action_dict[key] for key in epoch]) * DSCALE
        deviation = (BOPT - (BINIT + sum(henergy) - total_consumed_energy))

        #Objective function is to minimize the deviation from optimal battery level
        #Create a variable t such that |deviation|<=t
        #We tolerate a deviation of maximum 50 mWh
        # -t <= deviation <= t
        t = pulp.LpVariable('t', lowBound=50, cat='Continuous')
        model += t

        #Constraints A
        model += deviation <= t
        model += deviation >= -t

        #Constraints B
        #Create a dummy list of lists with entries [[epoch_1], [epoch_1, epoch_2], .... ]
        dummy = [epoch[0:i] for i in range(1,len(epoch)+1)]

        #dictionary of cumulative action variables [[a1], [a1 + a2],....]
        a_var_cum = {}

        #dictionary of cumulative harvested energy constants [[h1], [h1 + h2],....]
        henergy_cum = {} 

        for i in range(0 , len(epoch)):
            a_var_cum[epoch[i]] = pulp.lpSum([action_dict[key]*DSCALE for key in dummy[i]])
            henergy_cum[epoch[i]] = sum([henergy_dict[key] for key in dummy[i]])
            #henergy_cum = dict(zip(epoch, np.add.accumulate(henergy)))


        for key in epoch:
            model += BINIT + henergy_cum[key] - a_var_cum[key] <= BMAX
            model += BINIT + henergy_cum[key] - a_var_cum[key] >= BMIN

        #Solve the model
        optimization_flag = model.solve()
        #print(pulp.LpStatus[model.status])

    #Create list of optimized actions
    opt_act = {}
    if optimization_flag == 0:
        a_val = [DMIN]*SLOTS
    elif optimization_flag == 1:
        for var in epoch:
            opt_act[var] = action_dict[var].varValue
            a_val = list(opt_act.values())
    elif optimization_flag == 2:
        a_val = [DMAX]*SLOTS

    return a_val