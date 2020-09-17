import unittest
import pytest

import gym
from gym import envs

import gym_adserver
from gym_adserver.envs.ad import Ad

def test_init():
    ad = Ad(3, 2, 1)
    assert ad.id == '3'
    assert ad.impressions == 2
    assert ad.clicks == 1

def test_str():
    assert str(Ad(1, 100, 25)) == 'Ad: 1, CTR: 0.2500'

def test_repr():
    assert repr(Ad(1, 100, 25)) == '(25/100)'

@pytest.mark.parametrize("impressions,clicks,expected", [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 1),
    (100, 1, 0.01)
])
def test_ctr(impressions, clicks, expected):
    assert Ad(1, impressions, clicks).ctr() == expected

@pytest.mark.parametrize("id,impressions,clicks", [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (0, 1, 0),
    (0, 1, 1),
    (0, 1, 1)    
])
def test_eq(id, impressions, clicks):
    a = Ad(id, impressions, clicks) 
    b = Ad(id, impressions, clicks)
    assert a is not b
    assert a == b

@pytest.mark.parametrize("id,impressions,clicks", [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (0, 1, 0),
    (0, 1, 1),
    (0, 1, 1)
])
def test_not_eq(id, impressions, clicks):
    assert Ad(id, impressions, clicks) != Ad(0, 0, 0)