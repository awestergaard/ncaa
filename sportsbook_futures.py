'''
Created on Apr 7, 2017

@author: Adam
'''

from bs4 import BeautifulSoup
import urllib2
import numpy
from scipy import optimize

def better_prod(x, raw_prob):
    return (-x*numpy.sqrt(x*x-4*raw_prob**2+4*raw_prob)+x*x+2*raw_prob)/(2*(x*x+1))

def obj_fun(x, raw_prob):
    mod_prob = better_prod(x, raw_prob)
    return 1.-mod_prob.sum()

sportsbook = urllib2.urlopen(r'https://www.sportsbook.ag/sbk/sportsbook4/golf-the-masters-betting/the-masters-live-odds.sbk')
sportsbook_page = sportsbook.read()
sportsbook_soup = BeautifulSoup(sportsbook_page, 'html.parser')
items = sportsbook_soup.find_all('div', {'class' : 'clearfix row eventrow'})
names = [None]*len(items)
markets = numpy.array([0.]*len(items))
for i,item in enumerate(items):
    names[i] = item.find('span').text
    markets[i] = float(item.find('div', {'class' : 'market'}).text)
    
raw_prob = 100./(100.+markets)
x = optimize.brentq(f=obj_fun,a=0.,b=1.,args=(raw_prob,))