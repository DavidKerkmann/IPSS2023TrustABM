import random
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile, pstats, io
from pstats import SortKey
import pandas as pd

# define different infection states in a dictionary
infection_states = {'susceptible': 0, 'infected_asymptomatic': 1,
                    'infected_symptomatic': 2, 'recovered': 3, 'dead': 4}
vaccination_states = {'not vaccinated': False, 'vaccinated': True}
location_types = {'cemetery': 0, 'home': 1, 'work': 2}  # cemetery has to be 0
# truth institutions
truth_institutions = {'science': 0, 'government': 1, 'leader': 2}
# decisions
decisions = {'vaccinate': 0}

def opinion_assimilation(agents_in_location):
    avarage_trust = np.mean([a.trust_values for a in agents_in_location], axis = 0)
    for a in agents_in_location:
        a.trust_delta = avarage_trust - a.trust_values

def opinion_repulsion(a, agents_in_location):
    avarage_trust = np.mean([a.trust_values for a in agents_in_location], axis = 0)
    for a in agents_in_location:
        a.trust_delta = -avarage_trust + a.trust_values

def pandemic_influence(world, a, t):
    trust_delta = np.zeros(len(truth_institutions))
    # people gain trust in the government if the reproduction number is low
    trust_delta[truth_institutions['government']] += 1 - world.R
    # people gain trust in science if the reproduction number is high, but more at the start of the pandemic
    trust_delta[truth_institutions['science']] += world.R*np.exp(-t/24/7)
    return trust_delta

## Define parameters
pr = cProfile.Profile()
pr.enable()
# Set user parameters here
dt = 8
max_dt = 8
if dt>max_dt:
    raise Exception('dt>max_dt. Please change!')
n_days = 42 # days to simulate
t_0 = 0
number_of_ensamble_runs = 10
number_of_agents = 300
transmission_parameter = 1
trust_change_velocity = 0.01/max_dt # between 0 and 1, should be relatively low
threshold_influence = 0.05 # should be relatively low

influence_response_function = opinion_assimilation
behavior_bridge = pandemic_influence
time_in_asymptomatic = 4*24
time_in_symptomatic = 4*24
infected_percentage = 0.02 # initial viral distribution
# Set perceived truth values and thresholds
truths = np.zeros([len(decisions), len(truth_institutions)])
thresholds = np.zeros(len(decisions))
truths[decisions['vaccinate'],truth_institutions['science']] = 0.8
truths[decisions['vaccinate'],truth_institutions['government']] = 0.7
truths[decisions['vaccinate'],truth_institutions['leader']] = 0.0
thresholds[decisions['vaccinate']] = 1.5

#help variables
t_end = 24*n_days
ts = range(0, t_end, dt)


## Core implementation:

class Agent:
    def __init__(self, id, home_id, work_id, initial_infection_state, state_time, initial_vaccination_state, trust_values, truth_values) -> None:
        self.id = id
        self.home_id = home_id
        self.work_id = work_id
        self.infection_state = initial_infection_state
        self.state_time = state_time
        self.vaccination_state = initial_vaccination_state
        self.trust_values = trust_values
        self.trust_delta = [0.0]*len(truth_institutions)
        self.truth_values = truth_values
        self.leader_id = id

    def vaccinate(self):
        self.vaccination_state = vaccination_states['vaccinated']

    def set_infection_state(self, new_state):
        self.infection_state = new_state

class Location:
    def __init__(self, id, location_type, capacity=100) -> None:
        self.id = id
        self.location_type = location_type
        self.capacity = capacity

class World:
    def __init__(self, agents, locations, truths, thresholds) -> None:
        self.agents = agents
        self.locations = locations
        self.agentlocation = {}
        self.truths = truths
        self.thresholds = thresholds
        for a in self.agents:  # initialize agent locations in home
            self.agentlocation[a.id] = a.home_id
        self.new_cases = 0
        self.active_cases = len([a for a in self.agents if a.infection_state is infection_states['infected_symptomatic']])
        self.deaths = 0
        self.R = 0

    def move(self, t, dt):
        # go to work and home
        for a in self.agents:
            if(a.infection_state == infection_states['dead']):
                self.agentlocation[a.id] = location_types['cemetery']
            elif(a.infection_state == infection_states['infected_symptomatic']):
                # stay at home when symptomatic
                self.agentlocation[a.id] = a.home_id
            else:
                # go to work between 9am and 5pm
                if (t % 24 > 9 and t % 24 < 17):
                    self.agentlocation[a.id] = a.work_id
                else:
                    self.agentlocation[a.id] = a.home_id

    def update_infection_state(self, t, dt):
        for a in self.agents:
            a.state_time += dt  # update time in state
            if a.infection_state == infection_states['infected_asymptomatic']:
                if a.state_time > time_in_asymptomatic:
                    if random.random() > 0.5:
                        a.infection_state = infection_states['infected_symptomatic']
                        self.new_cases += 1
                    else:
                        a.infection_state = infection_states['recovered']
                    a.state_time = 0
                else:
                    pass
            elif a.infection_state == infection_states['infected_symptomatic']:
                if a.state_time > time_in_symptomatic:
                    if random.random() > 0.2:
                        a.infection_state = infection_states['recovered']
                    else:
                        a.infection_state = infection_states['dead']
                    a.state_time = 0
                else:
                    pass
            else:
                pass

    def infect(self, t, dt) -> None:
        for l in self.locations:
            agents_in_location = [
                a for a in self.agents if self.agentlocation[a.id] is l.id]
            if len(agents_in_location) > 0:
                sus = [a for a in agents_in_location if a.infection_state is
                       infection_states['susceptible']]
                inf = [a for a in agents_in_location if a.infection_state is
                       infection_states['infected_asymptomatic'] or a.infection_state is infection_states['infected_symptomatic']]
                # very sophisticated transmission model:
                # transmission probability is proportional to the number of infected people in the location
                # and the time spent there
                transmission_prob = transmission_parameter*len(inf)/len(agents_in_location)*dt/24
                for s in sus:
                    if random.random() < transmission_prob and s.vaccination_state is False:
                        s.set_infection_state(
                            infection_states['infected_asymptomatic'])
                        s.state_time = 0

    def agent_get_truth_values(self, t, dt):
        for a in self.agents:
            for d in decisions.values():
                for ti in truth_institutions.values():
                    if t is truth_institutions['science']:
                        a.truth_values[d,ti] = truths[d,ti]
                    if t is truth_institutions['government']:
                        a.truth_values[d,ti] = truths[d,ti]
                    if t is truth_institutions['leader']:
                        a.truth_values[d,ti] = self.agents[a.leader_id].truth_values[d,ti]

    def calc_decision_values(self, trust, truth):
        return np.dot(trust, truth)

    def agent_decisions(self, t, dt):
        for a in self.agents:
            if a.infection_state is not infection_states['dead']:
                for d in decisions.values():
                    if d == decisions['vaccinate']:
                        vac_decision_value = self.calc_decision_values(a.trust_values, a.truth_values[d,:])
                        vac_threshold = self.thresholds[d]/(1+threshold_influence*self.R+1e-15)
                        if vac_decision_value > vac_threshold:
                            a.vaccinate()
                            a.infection_state = infection_states['recovered']
           
    def update_trust(self, t, dt):
        for l in self.locations:
            agents_in_location = [
                a for a in self.agents if self.agentlocation[a.id] is l.id]
            if len(agents_in_location) > 1:
                influence_response_function(agents_in_location)
            
        for a in self.agents:
            if a.infection_state is not infection_states['dead']:
                a.trust_delta += behavior_bridge(self, a, t)
                a.trust_values += trust_change_velocity*dt/max_dt*a.trust_delta
                a.trust_values = [max(0.0, v) for v in a.trust_values]
                a.trust_values = [min(1.0, v) for v in a.trust_values]
                #a.trust_values = a.trust_values/np.sum(a.trust_values)

    def update_pandemic_statistics(self, t, dt):
        if (t//24 != (t+dt)//24):
            # update only every day
            if self.active_cases == 0:
                self.R = 0
            else:
                 # factor (1+time_in_asymptomatic/time_in_symptomatic) to account for dark figure
                self.R = self.new_cases/((1+time_in_asymptomatic/time_in_symptomatic)*self.active_cases)*(time_in_asymptomatic+time_in_symptomatic)/24
            self.active_cases = len([a for a in self.agents if a.infection_state == infection_states['infected_symptomatic']])
            self.new_cases = 0

    def step(self, t, dt):
        self.move(t, dt)
        self.update_infection_state(t, dt)
        self.infect(t, dt)
        self.agent_decisions(t,dt)
        self.update_trust(t, dt)
        self.agent_get_truth_values(t, dt)
        self.update_pandemic_statistics(t, dt)

# end of World class

# Helper Code to create workplaces for agents with enough capacity
def create_workplaces(n_agents, locations) -> None:
    # assign workplaces with random size between 5 and 60 agents and assign each agent a workplace
    while n_agents > 0:
        workplace_size = (int)(np.random.uniform(5, 30))
        locations.append(
            Location(len(locations), location_types['work'], workplace_size))
        n_agents = n_agents - workplace_size

# Helper code to create locations and agents+households.
def create_agents_and_homes(n_agents, locations, agents, infected_percentage):
    # for now, do random household size between 1 and 4 and a random work
    while n_agents > 0:
        hh_size = (int)(min(np.ceil(random.random()*4), n_agents))
        locations.append(Location(len(locations), location_types['home']))
        for _ in range(hh_size):
            work_ids = [l.id for l in locations if l.location_type ==
                location_types['work']]
            work_id = random.choice(work_ids)
            # use random trust distribution
            start_trusts = np.random.rand(3)
            #start_trusts = start_trusts/np.sum(start_trusts)
            start_truths = np.zeros((len(decisions), len(truth_institutions)))
            agent_infection_state = infection_states['susceptible']
            state_time = 0
            if random.random() < infected_percentage:
                if random.random() < 0.5:
                    agent_infection_state = infection_states['infected_asymptomatic']
                    state_time = random.random()*time_in_asymptomatic
                else:
                    agent_infection_state = infection_states['infected_symptomatic']
                    state_time = random.random()*time_in_symptomatic
            agents.append(Agent(len(agents), len(locations), work_id,
                          agent_infection_state, state_time, vaccination_states['not vaccinated'], start_trusts, start_truths))
        n_agents = n_agents - hh_size

    return locations, agents

# Code to help with saving and output
def save_infection_states(world, susceptible, infected, dead, recovered_vaccinated, vaccinated):
    susceptible.append(len(
        [a for a in world.agents if a.infection_state == infection_states['susceptible']]))
    infected.append(len([a for a in world.agents if a.infection_state == infection_states['infected_asymptomatic']
                    or a.infection_state == infection_states['infected_symptomatic']]))
    dead.append(len([a for a in world.agents if a.infection_state == infection_states['dead']]))
    recovered_vaccinated.append(len(
        [a for a in world.agents if a.infection_state == infection_states['recovered']]))
    vaccinated.append(len(
        [a for a in world.agents if a.vaccination_state is vaccination_states['vaccinated']]))
    return susceptible, infected, dead, recovered_vaccinated, vaccinated

def save_trust_values(trust_values, agents):
    trust_values.append([a.trust_values for a in agents])
    return trust_values

def plot_max_trust_values(ts, trust_values):
    trust_values = np.array(trust_values)
    #max trust values for each timestep for each insitution
    max_trust_value_each_timestep = np.zeros((trust_values.shape[0], trust_values.shape[2]))
    average_trust_values = np.sum(trust_values, axis = 1)/trust_values.shape[1]
    for i in range(trust_values.shape[2]):
        max_trust_value_each_timestep[:,i] = np.max(trust_values[:,:,i], axis=1)
    #plt.plot(ts, max_trust_value_each_timestep)
    plt.plot(ts, average_trust_values)
    plt.legend(['science', 'government', 'leader'])
    plt.xlabel('hours')
    plt.ylabel('trust')
    plt.show()

def plot_sir(ts, susceptible, infected, dead, recovered_vaccinated, vaccinated):
    plt.plot(ts, susceptible, ts, infected, ts, dead, ts, recovered_vaccinated, ts, vaccinated)
    plt.legend(['susceptible', 'infected', 'dead', 'recovered/vaccinated', 'vaccinated'])
    plt.xlabel('hours')
    plt.ylabel('number of agents')
    plt.show()

def plot_ensembles(ts, ensemble_vector, string_y_axis, legend):
    run_dim = ensemble_vector.shape[0]
    time_dim = ensemble_vector.shape[1]
    categories_dim = ensemble_vector.shape[2]
    ensemble_vector = np.array(ensemble_vector)
    
    trust_values_mean = ensemble_vector.mean(axis=0)
    trust_values_std = ensemble_vector.std(axis=0)
    
    plt.plot(ts, trust_values_mean)
    for i in range(categories_dim):
        plt.fill_between(ts, trust_values_mean[:,i]-trust_values_std[:,i], trust_values_mean[:,i]+trust_values_std[:,i], alpha = 0.1)
    plt.legend(legend, loc='lower center', bbox_to_anchor=(0.5, -0.8))
    plt.xlabel('hours')
    plt.ylabel(string_y_axis)
    return plt

def ensemble_runs():
    ensemble_trust_values = []
    ensemble_sir_states = []

    st = time.time()
    for _ in range(number_of_ensamble_runs):
        
        susceptible, infected, dead, recovered_vaccinated, vaccinated = [], [], [], [], []
        trust_values =[]
        #create world
        # initilize and create workplaces and agents
        locations = [Location(0, location_types['cemetery'])]  # cemetry is always 0
        agents = []
        create_workplaces(number_of_agents, locations)
        locations, agents = create_agents_and_homes(
            number_of_agents, locations, agents, infected_percentage)
        world = World(agents, locations, truths, thresholds)
        # run simulation
        t = t_0
        while t < t_end:
            susceptible, infected, dead, recovered_vaccinated, vaccinated = save_infection_states(world, susceptible, infected, dead, recovered_vaccinated, vaccinated)
            trust_values = save_trust_values(trust_values, agents)
            world.step(t, dt)
            t += dt       
        # save results for plot
        ensemble_trust_values.append(trust_values)
        ensemble_sir_states.append([susceptible, infected, dead, recovered_vaccinated, vaccinated])

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    return np.array(ensemble_sir_states), np.array(ensemble_trust_values)


# Main program
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

ensemble_sir_states, ensemble_trust_values = ensemble_runs()
# plot ensemble trust values
average_trust_values = ensemble_trust_values.mean(axis=2)
plt_trust= plot_ensembles(ts, average_trust_values, 'average trust', truth_institutions.keys())
plt_trust.show()

# plot ensemble sir states
# reshape and interchange axes
ensemble_sir_states=ensemble_sir_states.swapaxes(1,2)
plt_sir = plot_ensembles(ts, ensemble_sir_states, 'number of agents', ['susceptible', 'infected', 'dead', 'recovered/vaccinated', 'vaccinated'])
plt_sir.show()
#plot_sir(ts, susceptible, infected, dead, recovered_vaccinated, vaccinated)
x = 1

parameter_refinement = 20
parameter_range = np.linspace(0, 20*0.1/max_dt, parameter_refinement)
parameter_runs = []
for trust_change_velocity in parameter_range:
  ensemble_sir_states, ensemble_trust_values = ensemble_runs()
  parameter_runs.append(ensemble_sir_states[:,1,:].mean(axis=0)) # plot infected
plt.contourf(ts, parameter_range, parameter_runs, 20, cmap='RdGy')
plt.colorbar()

pr.disable()
sortby = SortKey.CUMULATIVE
s = io.StringIO()
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats(10)