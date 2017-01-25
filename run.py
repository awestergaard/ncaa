'''
Created on Jan 13, 2017

@author: Adam
'''

import multiprocessing
import numpy
import random
import urllib2

from scipy.stats import norm

from bs4 import BeautifulSoup

def rouletteWheel(weights):
    draw = random.uniform(0, sum(weights))
    pick = -1
    while draw > 0:
        pick += 1
        draw -= weights[pick]
    
    return pick

def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * numpy.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def ncdf(x):
    return 1. - 0.5*erfcc(x/(2**0.5))

def simulataion_data_to_csv(data):
    f = open(r'C:\kenpom_prob.csv', 'w')
    sorted_data = sorted(data.iteritems(),key=lambda x:-x[1])
    for k,v in sorted_data:
        if v > 0:
            f.write(k + ',' + str(v) + '\n')
    f.close()

class SimulationData():
    def populate(self):
        kenpom = urllib2.urlopen('http://kenpom.com/')
        kenpom_page = kenpom.read()
        kenpom_soup = BeautifulSoup(kenpom_page, 'html.parser')
        kenpom_table = kenpom_soup.find_all('tr')
        kenpom_team_names = []
        kenpom_adjo = []
        kenpom_adjd = []
        kenpom_adjt = []
        for row in kenpom_table[2:]:
            items = row.find_all('td')
            try:
                kenpom_adjo.append(float(items[5].get_text()))
                kenpom_adjd.append(float(items[7].get_text()))
                kenpom_adjt.append(float(items[9].get_text()))
                kenpom_team_names.append(items[1].get_text().strip().lower())
            except:
                pass

        bracketmatrix = urllib2.urlopen('http://bracketmatrix.com/')
        bracketmatrix_page = bracketmatrix.read()
        bracketmatrix_soup = BeautifulSoup(bracketmatrix_page, 'html.parser')
        bracketmatrix_table = bracketmatrix_soup.find_all('tr')
        self.team_names = []
        self.average_seeds = []
        self.adjo = []
        self.adjd = []
        self.adjt = []
        for row in bracketmatrix_table[3:]:
            items = row.find_all('td')
            try:
                average_seed = float(items[3].get_text())
                team_name = items[1].get_text().strip().lower()
                potential_kenpom_aliases = self.get_potential_kenpom_aliases(team_name)
                loc = -1
                for alias in potential_kenpom_aliases:
                    if alias in kenpom_team_names:
                        loc = kenpom_team_names.index(alias)
                        break
                
                if loc <> -1:
                    self.team_names.append(team_name)
                    self.average_seeds.append(float(average_seed))
                    self.adjo.append(kenpom_adjo[loc])
                    self.adjd.append(kenpom_adjd[loc])
                    self.adjt.append(kenpom_adjt[loc])
                
            except:
                pass
            
        self.averageo = numpy.average(kenpom_adjo)
        self.averaget = numpy.average(kenpom_adjo)

    @staticmethod
    def get_potential_kenpom_aliases(name):
        if name == "st. mary's (ca)":
            return ["saint mary's"]
        else:
            return [name,
                    name.replace('state', 'st.'),
                    ''.join([item[0] for item in name.split()])+'U']
    
    def simulate_tournament_round(self, seeding):
        round_adjo = [self.adjo[i] for i in seeding]
        round_adjd = [self.adjd[i] for i in seeding]
        round_adjt = [self.adjt[i] for i in seeding]
        n_teams = len(round_adjo)
        n_games = n_teams / 2
        tempos = [round_adjt[i] + round_adjt[n_teams-1-i] - self.averaget for i in xrange(n_games)]
        scoring_prob = [(round_adjo[i] + round_adjd[n_teams-1-i] - self.averageo)/200. for i in xrange(n_teams)]
        win_probability = [ncdf(numpy.sqrt(tempos[i]) *
                                (scoring_prob[i] - scoring_prob[n_teams-1-i]) /
                                 numpy.sqrt(scoring_prob[i]*(1-scoring_prob[i]) +
                                            scoring_prob[n_teams-1-i]*(1-scoring_prob[n_teams-1-i])))
                           for i in xrange(n_games)]
        outcomes = [random.random() for _ in xrange(n_games)]
        return [seeding[i] if outcomes[i] < win_probability[i] else seeding[n_teams-1-i] for i in xrange(n_games)]

    def simulate_tournament(self):
        average_seeds = self.average_seeds[:]
        locations = range(len(average_seeds))
        seeding = [0]*64
        for i in xrange(64):
            target_seed = i/4+1
            weights = [numpy.exp(-(x-target_seed)**2) for x in average_seeds]
            pick = rouletteWheel(weights)
            seeding[i] = locations[pick]
            del average_seeds[pick]
            del locations[pick]
            
        while len(seeding) > 1:
            seeding = self.simulate_tournament_round(seeding)
        
        return seeding[0]

    def run_simulation(self, n_trials=10000):
        winners = [None]*len(n_trials)
        for i in xrange(n_trials):
            winners[i] = self.simulate_tournament()
        results = [0]*len(self.team_names)
        for team in xrange(len(self.team_names)):
            results[team] = float(winners.count(team)) / n_trials
            
        return {team_name: result for team_name, result in zip(self.team_names, results)}

def simulate_tournament_round(seeding, data):
    round_adjo = [data['adjo'][i] for i in seeding]
    round_adjd = [data['adjd'][i] for i in seeding]
    round_adjt = [data['adjt'][i] for i in seeding]
    n_teams = len(round_adjo)
    n_games = n_teams / 2
    tempos = [round_adjt[i] + round_adjt[n_teams-1-i] - data['averaget'] for i in xrange(n_games)]
    scoring_prob = [(round_adjo[i] + round_adjd[n_teams-1-i] - data['averageo'])/200. for i in xrange(n_teams)]
    win_probability = [ncdf(numpy.sqrt(tempos[i]) *
                            (scoring_prob[i] - scoring_prob[n_teams-1-i]) /
                             numpy.sqrt(scoring_prob[i]*(1-scoring_prob[i]) +
                                        scoring_prob[n_teams-1-i]*(1-scoring_prob[n_teams-1-i])))
                       for i in xrange(n_games)]
    outcomes = [random.random() for _ in xrange(n_games)]
    return [seeding[i] if outcomes[i] < win_probability[i] else seeding[n_teams-1-i] for i in xrange(n_games)]

def simulate_tournament(data):
    average_seeds = data['average_seeds'][:]
    locations = range(len(average_seeds))
    seeding = [0]*64
    for i in xrange(64):
        target_seed = i/4+1
        weights = [numpy.exp(-(x-target_seed)**2) for x in average_seeds]
        pick = rouletteWheel(weights)
        seeding[i] = locations[pick]
        del average_seeds[pick]
        del locations[pick]
            
    while len(seeding) > 1:
        seeding = simulate_tournament_round(seeding, data)
    
    return seeding[0]

def run_simulation(sim, n_trials=10000):
    pool = multiprocessing.Pool()
    data = {'average_seeds': sim.average_seeds,
            'adjo': sim.adjo,
            'adjd': sim.adjd,
            'adjt': sim.adjt,
            'averageo': sim.averageo,
            'averaget': sim.averaget}
    winners = pool.map(simulate_tournament, (data for _ in xrange(n_trials)))
    results = [0]*len(sim.team_names)
    for team in xrange(len(sim.team_names)):
        results[team] = float(winners.count(team)) / n_trials
        
    return {team_name: result for team_name, result in zip(sim.team_names, results)}

sim = SimulationData()
sim.populate()