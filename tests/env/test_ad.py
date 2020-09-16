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

def test_ctr():
    assert Ad(1, 0, 0).ctr() == 0
    assert Ad(1, 1, 0).ctr() == 0
    assert Ad(1, 1, 1).ctr() == 1
    assert Ad(1, 100, 1).ctr() == 0.01

def test_eq():
    assert Ad(1, 0, 0) == Ad(1, 0, 0)
    assert Ad(1, 1, 0) == Ad(1, 1, 0)
    assert Ad(1, 1, 1) == Ad(1, 1, 1)
    assert Ad(1, 0, 0) != Ad(0, 0, 0)
    assert Ad(0, 1, 0) != Ad(0, 0, 0)
    assert Ad(0, 0, 1) != Ad(0, 0, 0)

def test_str():
    assert str(Ad(1, 100, 25)) == 'Ad: 1, CTR: 0.2500'

def test_repr():
    assert repr(Ad(1, 100, 25)) == '(25/100)'