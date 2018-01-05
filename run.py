'''
Created on Jan 13, 2017

@author: Adam
'''

from http.cookiejar import CookieJar
from multiprocessing.pool import Pool
import numpy
import random
import urllib

team_mod = {}

bracket = ['', #1-1
           '', #1-2
           'gonzaga', #1-3
           'north carolina', #1-4
           '', #2-4
           '', #2-3
           '', #2-2
           '', #2-1
           '', #3-1
           'oregon', #3-2
           '', #3-3
           '', #3-4
           '', #4-4
           '', #4-3
           '', #4-2
           '', #4-1
           '', #5-1
           '', #5-2
           '', #5-3
           '', #5-4
           '', #6-4
           '', #6-3
           '', #6-2
           '', #6-1
           'south carolina', #7-1
           '', #7-2
           '', #7-3
           '', #7-4
           '', #8-4
           '', #8-3
           '', #8-2
           '', #8-1
           '', #9-1
           '', #9-2
           '', #9-3
           '', #9-4
           '', #10-4
           '', #10-3
           '', #10-2
           '', #10-1
           '', #11-1
           '', #11-2
           '', #11-3
           '', #11-4
           '', #12-4
           '', #12-3
           '', #12-2
           '', #12-1
           '', #13-1
           '', #13-2
           '', #13-3
           '', #13-4
           '', #14-4
           '', #14-3
           '', #14-2
           '', #14-1
           '', #15-1
           '', #15-2
           '', #15-3
           '', #15-4
           '', #16-4
           '', #16-3
           '', #16-2
           '' #16-1
           ]

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
    sorted_data = sorted(data.items(),key=lambda x:-x[1])
    for k,v in sorted_data:
        if v > 0:
            f.write(k + ',' + str(v) + '\n')
    f.close()

class SimulationData():
    def populate(self):
        kenpom = urllib.request.urlopen('http://kenpom.com/')
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
                kenpom_team_name = items[1].get_text().strip().lower().strip(';')
                for i in range(17):
                    kenpom_team_name = kenpom_team_name.strip(' ' + str(i))
                kenpom_team_names.append(kenpom_team_name)
                
            except:
                pass
    
        cj = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

        authentication_url = 'http://kenpom.com/handlers/login_handler.php'
        authentication_payload = {
            'email': 'a.westergaard@gmail.com',
            'password': 'ey6qRMvgcj'
        }
        authentication_data = urllib.parse.urlencode(authentication_payload)
        authentication_binary_data = authentication_data.encode('UTF-8')
        opener.open(authentication_url, authentication_binary_data)

        kenpom_stats = opener.open('http://kenpom.com/stats.php')
        kenpom_stats_page = kenpom_stats.read()
        kenpom_stats_soup = BeautifulSoup(kenpom_stats_page, 'html.parser')
        kenpom_stats_table = kenpom_stats_soup.find_all('tr')
        kenpom_to_o = [None]*len(kenpom_team_names)
        kenpom_or_o = [None]*len(kenpom_team_names)
        kenpom_ftrate_o = [None]*len(kenpom_team_names)
        kenpom_to_d = [None]*len(kenpom_team_names)
        kenpom_or_d = [None]*len(kenpom_team_names)
        kenpom_ftrate_d = [None]*len(kenpom_team_names)
        for row in kenpom_stats_table[2:]:
            items = row.find_all('td')
            try:
                name = items[0].get_text().strip().lower()
                loc = kenpom_team_names.index(name)
                kenpom_to_o[loc] = float(items[8].get_text())
                kenpom_or_o[loc] = float(items[10].get_text())
                kenpom_ftrate_o[loc] = float(items[12].get_text())
                kenpom_to_d[loc] = float(items[18].get_text())
                kenpom_or_d[loc] = float(items[20].get_text())
                kenpom_ftrate_d[loc] = float(items[22].get_text())
            except:
                pass
            
        kenpom_teamstats = opener.open('http://kenpom.com/teamstats.php')
        kenpom_teamstats_page = kenpom_teamstats.read()
        kenpom_teamstats_soup = BeautifulSoup(kenpom_teamstats_page, 'html.parser')
        kenpom_teamstats_table = kenpom_teamstats_soup.find_all('tr')
        kenpom_3p_o = [None]*len(kenpom_team_names)
        kenpom_2p_o = [None]*len(kenpom_team_names)
        kenpom_ft_o = [None]*len(kenpom_team_names)
        kenpom_3pa_o = [None]*len(kenpom_team_names)
        for row in kenpom_teamstats_table[2:]:
            items = row.find_all('td')
            try:
                name = items[0].get_text().strip().lower()
                loc = kenpom_team_names.index(name)
                kenpom_3p_o[loc] = float(items[2].get_text())
                kenpom_2p_o[loc] = float(items[4].get_text())
                kenpom_ft_o[loc] = float(items[6].get_text())
                kenpom_3pa_o[loc] = float(items[14].get_text())
            except:
                pass
            
        kenpom_teamstats_d = opener.open('http://kenpom.com/teamstats.php?od=d')
        kenpom_teamstats_d_page = kenpom_teamstats_d.read()
        kenpom_teamstats_d_soup = BeautifulSoup(kenpom_teamstats_d_page, 'html.parser')
        kenpom_teamstats_d_table = kenpom_teamstats_d_soup.find_all('tr')
        kenpom_3p_d = [None]*len(kenpom_team_names)
        kenpom_2p_d = [None]*len(kenpom_team_names)
        kenpom_ft_d = [None]*len(kenpom_team_names)
        kenpom_3pa_d = [None]*len(kenpom_team_names)
        for row in kenpom_teamstats_d_table[2:]:
            items = row.find_all('td')
            try:
                name = items[0].get_text().strip().lower()
                loc = kenpom_team_names.index(name)
                kenpom_3p_d[loc] = float(items[2].get_text())
                kenpom_2p_d[loc] = float(items[4].get_text())
                kenpom_ft_d[loc] = float(items[6].get_text())
                kenpom_3pa_d[loc] = float(items[14].get_text())
            except:
                pass

        bracketmatrix = urllib.request.urlopen('http://bracketmatrix.com/')
        bracketmatrix_page = bracketmatrix.read()
        bracketmatrix_soup = BeautifulSoup(bracketmatrix_page, 'html.parser')
        bracketmatrix_table = bracketmatrix_soup.find_all('tr')
        self.team_names = ['']
        self.average_seeds = [0]
        self.num_brackets = [0]
        self.adjo = [50]
        self.adjd = [150.]
        self.adjt = [1000]
        self.to_o = [0]
        self.or_o = [0]
        self.ftrate_o = [0]
        self.to_d = [0]
        self.or_d = [0]
        self.ftrate_d = [0]
        self._3p_o = [0]
        self._2p_o = [0]
        self.ft_o = [0]
        self._3pa_o = [0]
        self._3p_d = [0]
        self._2p_d = [0]
        self.ft_d = [0]
        self._3pa_d = [0]
        for row in bracketmatrix_table[3:]:
            items = row.find_all('td')
            try:
                average_seed = float(items[3].get_text())
                num_brackets = int(items[4].get_text())
                team_name = items[1].get_text().strip().lower()
                #print(team_name)
                potential_kenpom_aliases = self.get_potential_kenpom_aliases(team_name)
                loc = -1
                for alias in potential_kenpom_aliases:
                    if alias in kenpom_team_names:
                        loc = kenpom_team_names.index(alias)
                        break
                
                if loc != -1:
                    if team_name in team_mod:
                        print('modifying ' + team_name)
                        kenpom_adjo[loc] += team_mod[team_name]
                        kenpom_adjd[loc] -= team_mod[team_name]
                        
                    self.team_names.append(team_name)
                    self.average_seeds.append(average_seed)
                    self.num_brackets.append(num_brackets)
                    self.adjo.append(kenpom_adjo[loc])
                    self.adjd.append(kenpom_adjd[loc])
                    self.adjt.append(kenpom_adjt[loc])
                    self.to_o.append(kenpom_to_o[loc])
                    self.or_o.append(kenpom_or_o[loc])
                    self.ftrate_o.append(kenpom_ftrate_o[loc])
                    self.to_d.append(kenpom_to_d[loc])
                    self.or_d.append(kenpom_or_d[loc])
                    self.ftrate_d.append(kenpom_ftrate_d[loc])
                    self._3p_o.append(kenpom_3p_o[loc])
                    self._2p_o.append(kenpom_2p_o[loc])
                    self.ft_o.append(kenpom_ft_o[loc])
                    self._3pa_o.append(kenpom_3pa_o[loc])
                    self._3p_d.append(kenpom_3p_d[loc])
                    self._2p_d.append(kenpom_2p_d[loc])
                    self.ft_d.append(kenpom_ft_d[loc])
                    self._3pa_d.append(kenpom_3pa_d[loc])
            except:
                pass
            
        self.averageo = numpy.average(kenpom_adjo)
        self.averaget = numpy.average(kenpom_adjt)
        self.max_brackets = max(self.num_brackets)

    @staticmethod
    def get_potential_kenpom_aliases(name):
        if name == "st. mary's (ca)":
            return ["saint mary's"]
        elif name == "miami (fla.)":
            return ["miami fl"]
        elif name == "nc-wilmington":
            return ["unc wilmington"]
        elif name == "e. tennessee state":
            return ["east tennessee st."]
        else:
            return [name,
                    name.replace('state', 'st.'),
                    ''.join([item[0] for item in name.split()])+'u']

def win_probability(team1_index, team2_index, data):
    team1_adjo = data['adjo'][team1_index]
    team1_adjd = data['adjd'][team1_index]
    team1_adjt = data['adjo'][team1_index]
    team2_adjo = data['adjo'][team2_index]
    team2_adjd = data['adjd'][team2_index]
    team2_adjt = data['adjt'][team2_index]
    
    game_tempo = team1_adjt + team2_adjt - data['averaget']
    team1_scoring_prob = (team1_adjo + team2_adjd - data['averageo']) / 200.
    team2_scoring_prob = (team2_adjo + team1_adjd - data['averageo']) / 200.
    return ncdf(numpy.sqrt(game_tempo) * (team1_scoring_prob - team2_scoring_prob) /
                numpy.sqrt(team1_scoring_prob*(1.-team1_scoring_prob) +
                           team2_scoring_prob*(1.-team2_scoring_prob)))

def simulate_tournament_round(seeding, data):
    round_adjo = [data['adjo'][i] for i in seeding]
    round_adjd = [data['adjd'][i] for i in seeding]
    round_adjt = [data['adjt'][i] for i in seeding]
    n_teams = len(round_adjo)
    n_games = int(n_teams / 2)
    tempos = [round_adjt[i] + round_adjt[n_teams-1-i] - data['averaget'] for i in range(n_games)]
    scoring_prob = [(round_adjo[i] + round_adjd[n_teams-1-i] - data['averageo'])/200. for i in range(n_teams)]
    win_probability = [ncdf(numpy.sqrt(tempos[i]) *
                            (scoring_prob[i] - scoring_prob[n_teams-1-i]) /
                             numpy.sqrt(scoring_prob[i]*(1-scoring_prob[i]) +
                                        scoring_prob[n_teams-1-i]*(1-scoring_prob[n_teams-1-i])))
                       for i in range(n_games)]
    outcomes = [random.random() for _ in range(n_games)]
    return [seeding[i] if outcomes[i] < win_probability[i] else seeding[n_teams-1-i] for i in range(n_games)]

def pick_seeds(data):
    average_seeds = data['average_seeds'][:]
    percent_of_brackets = [num_brackets / float(data['max_brackets']) for num_brackets in data['num_brackets']]
    locations = range(len(average_seeds))
    seeding = [0]*64
    for i in range(64):
        target_seed = i/4+1
        weights = [numpy.exp(-x*x/n)
                + numpy.exp(-(x-target_seed)*(x-target_seed))*n for x, n in zip(average_seeds, percent_of_brackets)]
        pick = rouletteWheel(weights)
        seeding[i] = locations[pick]
        del average_seeds[pick]
        del locations[pick]
        
    return seeding

def pick_seeds2(data):
    seeding = []
    for i in range(10):
        seeding += random.sample(range(4*i, 4*(i+1)),4)
        
    seeding += random.sample(range(41,53),8)
    
    for i in range(13,16):
        seeding += random.sample(range(4*i, 4*(i+1)),4)
        
    seeding += random.sample(range(64,72),4)
    
    return seeding

def pick_seeds3(data):
    seeding = [None]*35
    up = list(range(5))
    for i in range(35):
        if i-2 in up:
            seeding[i] = i-2
            del up[up.index(i-2)]
        else:
            random.shuffle(up)
            seeding[i] = up[0]
            del up[0]
        up.append(i+5)
    seeding[35:40] += up
    seeding[41:49] += random.sample(range(41,53),8)
    for i in range(13,16):
        seeding += random.sample(range(4*i, 4*(i+1)),4)
    seeding += random.sample(range(64,72),4)
    
    return seeding

def pick_seed_final(data):
    seeding = [None]*64
    for i in range(64):
        if len(bracket[i]) == 2:
            choice = random.choice(bracket[i])
            seeding[i] = data['team_names'].index(choice)
        else:
            seeding[i] = data['team_names'].index(bracket[i])
            
    return seeding

def simulate_tournament(data):
    #seeding = pick_seeds(data)
    #seeding = pick_seeds2(data)
    seeding = pick_seeds3(data)
    #seeding = pick_seed_final(data)
            
    while len(seeding) > 1:
        seeding = simulate_tournament_round(seeding, data)
    
    return seeding[0]

def get_raw_data_from_sim(sim):
    return {'team_names': sim.team_names,
            'average_seeds': sim.average_seeds,
            'num_brackets': sim.num_brackets,
            'adjo': sim.adjo,
            'adjd': sim.adjd,
            'adjt': sim.adjt,
            'averageo': sim.averageo,
            'averaget': sim.averaget,
            'max_brackets': sim.max_brackets}

def run_simulation(sim, n_trials=10000):
    sim_pool = Pool()
    data = get_raw_data_from_sim(sim)
    winners = sim_pool.map(simulate_tournament, (data for _ in range(n_trials)))
    results = [0]*len(sim.team_names)
    for team in range(len(sim.team_names)):
        results[team] = float(winners.count(team)) / n_trials
        
    return {team_name: result for team_name, result in zip(sim.team_names, results)}

def simulate_picks(sim, n_trials=10000):
    sim_pool = Pool()
    data = {'average_seeds': sim.average_seeds,
            'num_brackets': sim.num_brackets,
            'adjo': sim.adjo,
            'adjd': sim.adjd,
            'adjt': sim.adjt,
            'averageo': sim.averageo,
            'averaget': sim.averaget,
            'max_brackets': sim.max_brackets}
    seedings = sim_pool.map(pick_seeds, (data for _ in range(n_trials)))
    results = [(-1.,0.)]*len(sim.team_names)
    for team in range(len(sim.team_names)):
        team_seeds = [seedings[i].index(team) for i in range(n_trials) if team in seedings[i]]
        if len(team_seeds):
            results[team] = (numpy.min(team_seeds)/4+1, 
                             numpy.average(team_seeds)/4.+1., 
                             numpy.max(team_seeds)/4+1,
                             float(len(team_seeds)) / n_trials)
        
    return {team_name: result for team_name, result in zip(sim.team_names, results)}

def run():
    sim = SimulationData()
    print('populating sim')
    sim.populate()
    print('starting sim')
    import datetime
    start_time = datetime.datetime.now()
    results = run_simulation(sim, n_trials=100000)
    end_time = datetime.datetime.now()
    print('sim done after ' + str(end_time-start_time))
    simulataion_data_to_csv(results)
    print('done')