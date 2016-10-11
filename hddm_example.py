#-----------------------------------------------------------------------------------------------------
# Author: David Greatrex, University of Cambridge.  
# Date: 11/10/2016
# Name: Example of fitting a hierarchical drift diffusion model to human forced choice decision-making data.
# Requirements: The script contains python code that should be run in an iPython notebook.
# Cell 2 and 3 require 7 parallel processes to be run on the machine separately which you need to initiate
# using a separate terminal window using the commented code in cell 3.
#-----------------------------------------------------------------------------------------------------

#-------------------------------------
# CELL 1
#-------------------------------------
import hddm
import numpy as np
import pandas as pd
import os as os
%pylab inline
from pylab import *
%matplotlib inline
import matplotlib.pyplot as plt
import os
main_folder = '.../hddm_example'

#-------------------------------------
# CELL 2
#-------------------------------------
# Tests all combinations of letting v, a, t_er vary by periodicity condition.
# Allows v, vary across difficulty condition.
# Group level estimate of trial by trial variability.
# z is calcualted for each P.       
def run_model_set(id):
    
    # packages
    import hddm
    # define folders & load data
    data_folder = [main_folder + "hddm_data/"]
    trace_folder = '/model_trace'
    filename = "hddm_stim_coding.csv"
    data = hddm.load_csv(data_folder + filename)
    # set sampling criteria
    no_samples = 15000
    burn_in = 5000
    
    # run parallel processes
    if id == 0: 
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'a': ['periodicity'], 'v': ['level']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)
        return m
    
    if id == 1:
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'v': ['periodicity', 'level']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)
        return m 
        
    if id == 2:
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'v': ['level'], 't': ['periodicity']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)
        return m  

    if id == 3:
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'v': ['periodicity', 'level'], 't': ['periodicity']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)      
        return m 

    if id == 4:
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'a': ['periodicity'], 'v': ['level'], 't': ['periodicity']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)  
        return m 
      
    if id == 5:
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'a': ['periodicity'], 'v': ['periodicity', 'level']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)   
        return m  
        
    if id == 6: 
        print('running model %i'%id);
        m = hddm.HDDM(data, include=('z', 'sv', 'st', 'sz'), group_only_nodes=['sv', 'st', 'sz'], 
                  depends_on={'a': ['periodicity'], 'v': ['periodicity', 'level'], 't': ['periodicity']}, p_outlier=0.05)
        m.find_starting_values()
        m.sample(no_samples, burn=burn_in, dbname=main_folder + trace_folder + '/db%i'%id, db='pickle')
        m.save(main_folder + trace_folder + '/set_m%i%i'%id)   
        return m

#-------------------------------------
# CELL 3
#-------------------------------------
#----------------------
# RUN MODEL SET 1
#----------------------
# start 7 CPU clusters in background - enter into new terminal.
# ipcluster start -n 7

# run model set 1
from IPython.parallel import Client
v = Client()[:]
jobs = v.map(run_model_set, range(7)) # 7 is the number of live CPUs required
m_set = jobs.get()

# stop 7 CPU clusters in background - enter into new terminal.
# ipcluster stop

#-------------------------------------
# CELL 4
#-------------------------------------
#----------------------
# Load models from file
#----------------------
model_1 = hddm.load(main_folder + trace_folder + '/set_m0')
model_2 = hddm.load(main_folder + trace_folder + '/set_m1')
model_3 = hddm.load(main_folder + trace_folder + '/set_m2')
model_4 = hddm.load(main_folder + trace_folder + '/set_m3')
model_5 = hddm.load(main_folder + trace_folder + '/set_m4')
model_6 = hddm.load(main_folder + trace_folder + '/set_m5')
model_7 = hddm.load(main_folder + trace_folder + '/set_m6')
model_list = [model_1,model_2,model_3,model_4,model_5,model_6,model_7]

#-------------------------------------
# CELL 5
#-------------------------------------
#----------------------
# Extract ordered DIC values to .csv file
#----------------------
dic_folder = '/DIC'
dic_array = []
for i in range(0,len(model_list)):
    dic_array.append(model_list[i].dic)
model_id = list(range(1,len(model_list)+1))
dic_table = pd.DataFrame({'model_id' : model_id,'DIC' : dic_array})
# sort tables by values
dic_table = dic_table.sort_values(by = ['DIC'])
# save to file
dic_table.to_csv(main_folder + dic_folder + '/diffusion_set_DIC.csv', index=False)

#-------------------------------------
# CELL 6
#-------------------------------------
#----------------------
# Extract model statistics and save to .csv file
#----------------------
statistics_folder = '/statistics'
for i in range(0,len(model_list)):
    stats = model_list[i].gen_stats()
    filename = main_folder + statistics_folder + '/model_' + str(i+1) + '/model_' + str(i+1) + '_stats.csv'
    os.makedirs(os.path.dirname(filename))
    stats.to_csv(filename, index=True, index_label='Parameter')

#-------------------------------------
# CELL 7
#-------------------------------------
#----------------------
# Save posterior plots to file for each model
#----------------------
plots_folder = '/statistics'
pwd = os.getcwd()
for i in range(0,len(model_list)):
    filename =  main_folder + plots_folder + '/model_' + str(i+1) + '/'
    os.makedirs(os.path.dirname(filename))
    os.chdir(filename)
    model_list[i].plot_posteriors(save=True)
    plt.clf()
    print('finished plotting posterior plots for model ' + str(i+1))

#-------------------------------------
# CELL 8
#-------------------------------------
#----------------------
# Test whether the models have convered using the Geweke statistic
# You can also use the R-hat statistic (Gelman-Rubin) described here: http://ski.clps.brown.edu/hddm_docs/howto.html
#----------------------
from kabuki.analyze import check_geweke
for i in range(0,len(model_list)):
    print '-' * 30
    print check_geweke(model_list[i],assert_=False) 

#-------------------------------------
# CELL 9
#-------------------------------------
#----------------------
# Select one of the models for comparing parameter estimates: Pick the full model for demonstration.
#----------------------
tmp = model_list[6]

#-------------------------------------
# CELL 10
#-------------------------------------
#----------------------
# Plot boundary conditions
#----------------------
a_AP, a_P = tmp.nodes_db.node[['a(Aperiodic)', 'a(Periodic)']]
hddm.analyze.plot_posterior_nodes([a_AP, a_P])
plt.xlabel('Boundary condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of boundary condition group means')
#plt.savefig('hddm_demo_fig_06.pdf')
print "P(AP > P) = ", (a_AP.trace() > a_P.trace()).mean()

#-------------------------------------
# CELL 11
#-------------------------------------
#----------------------
# Plot drift rate conditions
#----------------------
APvneg4,Pvneg4 = tmp.nodes_db.node[['v(-4.Aperiodic)','v(-4.Periodic)']]
hddm.analyze.plot_posterior_nodes([APvneg4,Pvneg4])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "Neg 4 - P(AP > P) = ", (APvneg4.trace() > Pvneg4.trace()).mean()

APvneg2,Pvneg2 = tmp.nodes_db.node[['v(-2.Aperiodic)','v(-2.Periodic)']]
hddm.analyze.plot_posterior_nodes([APvneg2,Pvneg2])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "Neg 2 - P(AP > P) = ", (APvneg2.trace() > Pvneg2.trace()).mean()

APvneg1,Pvneg1 = tmp.nodes_db.node[['v(-1.Aperiodic)','v(-1.Periodic)']]
hddm.analyze.plot_posterior_nodes([APvneg1,Pvneg1])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "Neg 1 - P(AP > P) = ", (APvneg1.trace() > Pvneg1.trace()).mean()

APv0,Pv0 = tmp.nodes_db.node[['v(0.Aperiodic)','v(0.Periodic)']]
hddm.analyze.plot_posterior_nodes([APv0,Pv0])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "Zero - P(AP > P) = ", (APv0.trace() > Pv0.trace()).mean()

APv1,Pv1 = tmp.nodes_db.node[['v(1.Aperiodic)','v(1.Periodic)']]
hddm.analyze.plot_posterior_nodes([APv1,Pv1])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "1 - P(AP > P) = ", (APv1.trace() > Pv1.trace()).mean()

APv2,Pv2 = tmp.nodes_db.node[['v(2.Aperiodic)','v(2.Periodic)']]
hddm.analyze.plot_posterior_nodes([APvneg4,Pvneg4])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "2 - P(AP > P) = ", (APv2.trace() > Pv2.trace()).mean()

APv4,Pv4 = tmp.nodes_db.node[['v(4.Aperiodic)','v(4.Periodic)']]
hddm.analyze.plot_posterior_nodes([APv4,Pv4])
plt.xlabel('Drift rate condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift rate group means')
print "4 - P(AP > P) = ", (APv4.trace() > Pv4.trace()).mean()

#-------------------------------------
# CELL 12
#-------------------------------------
#----------------------
# Plot non-decision time
#----------------------
t_AP, t_P = tmp.nodes_db.node[['t(Aperiodic)', 't(Periodic)']]
hddm.analyze.plot_posterior_nodes([t_AP, t_P])
plt.xlabel('Non-decision time condition')
plt.ylabel('Posterior probability')
plt.title('Posterior of non-decision time condition group means')
print "P(AP > P) = ", (t_AP.trace() > t_P.trace()).mean()