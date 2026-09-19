import numpy as np
import pytest

from neuroterrarium.world import ACTION_DT, Body, Food, Obstacle, Stimulus, World, stream


def test_observation_local_and_idempotent():
    w=World(1,1); w.bodies=[Body(10,10,0)]; w.foods=[Food(16,10)]
    a=w.observe(); assert a.shape==(1,59)
    np.testing.assert_array_equal(a,w.observe())
    w2=World(1,1); w2.bodies=[Body(10,10,0)]; w2.foods=[Food(1000,1000)]
    w3=World(1,1); w3.bodies=[Body(10,10,0)]; w3.foods=[]
    np.testing.assert_array_equal(w2.observe(),w3.observe())


def test_food_occlusion_and_contact():
    w=World(1,1); w.bodies=[Body(10,10,0)]; w.foods=[Food(20,10)]
    w.obstacles=[Obstacle(15,10,2)]
    assert not w.observe()[0,48:53].any()
    for _ in range(10):w.advance([[0,0,0,1]])
    assert w.bodies[0].food==0


def test_food_competition_conserves_and_is_symmetric():
    w=World(1,2);w.bodies=[Body(9.4,10,0),Body(10.6,10,0)];w.foods=[Food(10,10,.001)]
    w.advance([[0,0,0,1],[0,0,0,1]])
    assert sum(b.food for b in w.bodies)==pytest.approx(.001)
    assert w.bodies[0].food==pytest.approx(w.bodies[1].food)
    assert w.foods[0].amount==0


def test_collision_and_action_limits():
    w=World(1,1);w.bodies=[Body(10,10,0)];w.obstacles=[Obstacle(12,10,1)]
    for _ in range(100):w.advance([[200,0,100,0]])
    assert w.bodies[0].x <= 10.45
    assert w.bodies[0].collisions>0
    assert max(w.bodies[0].action)<=1
    with pytest.raises(ValueError):w.advance([[float('nan')]*4])


def test_shadows_have_no_automatic_damage():
    w=World(1,1);w.bodies=[Body(10,10,0)];w.stimuli=[Stimulus(10,10,5,physical=False)]
    w.advance([[0,0,0,0]])
    assert w.bodies[0].energy==pytest.approx(1-.001*ACTION_DT)


def test_snapshot_continuation_and_panel_independence():
    w=World(52,3,'looming'); a=np.array([[.2,.3,.4,.5]]*3)
    for _ in range(5):w.advance(a)
    c=World.restore(w.snapshot())
    for _ in range(20):
        w.observe();w.observe();w.advance(a);c.advance(a)
    assert w.snapshot()==c.snapshot()


def test_projection_input_really_changes_when_stimulus_changes():
    w=World(1,1);w.bodies=[Body(10,10,0)];w.stimuli=[Stimulus(18,10,1,growth=10)]
    w.observe();w.advance([[0]*4]);a=w.observe()[0]
    assert a[32:48].max()>0
    w.stimuli=[];w.advance([[0]*4]);b=w.observe()[0]
    assert not b[16:32].any()
    assert not np.maximum(b[32:48],0).any()


def test_named_streams_independent():
    a=stream(10,'a');b=stream(10,'b');expected=stream(10,'b').random(20)
    a.random(100000);np.testing.assert_array_equal(b.random(20),expected)
