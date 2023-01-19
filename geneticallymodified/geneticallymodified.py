#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()


# In[2]:


from IPython.display import Audio


# In[3]:


rep_init = 2
repeat = 48
bar = 4
beat = 4
unit = 2
reso = beat*unit
prob_co = 0.2
prob_mu = 0.1

track = np.reshape(np.arange(bar*reso), [1, bar, reso])

def crossover(track, bar=bar, reso=reso):
    new_track = np.array(track)
    i0 = rng.choice(reso, size=[2, 1])
    ir = 1+rng.choice(reso)
    ii = (i0+np.arange(ir))%reso
    new_track[np.array([[0], [1]]), ii] = new_track[np.array([[0], [1]]), ii][::-1]
    return new_track

def identity(track, bar=bar, reso=reso):
    new_track = np.array(track)
    return new_track
def rotation(track, bar=bar, reso=reso):
    new_track = np.array(track)
    ii = np.sort(rng.choice(reso, size=2, replace=False))
    ii[1] += 1
    l = rng.integers(1, ii[1]-ii[0])
    new_track[ii[0]:ii[1]] = np.roll(new_track[ii[0]:ii[1]], l)
    return new_track
def inversion(track, bar=bar, reso=reso):
    new_track = np.array(track)
    ii = np.sort(rng.choice(reso, size=2, replace=False))
    ii[1] += 1
    l = rng.integers(1, ii[1]-ii[0])
    new_track[ii[0]:ii[1]] = new_track[ii[0]:ii[1]][::-1]
    return new_track
def overwrite(track, bar=bar, reso=reso):
    new_track = np.array(track)
    i0 = rng.choice(reso, size=[2, 1])
    ir = 1+rng.choice(reso)
    ii = (i0+np.arange(ir))%reso
    new_track[ii[1]] = new_track[ii[0]]
    return new_track

# methods = [identity]
methods = [rotation, inversion, overwrite]
def mutation(track, bar=bar, reso=reso, methods=methods):
    method = rng.choice(methods)
    new_track = method(track, bar=bar, reso=reso)
    return new_track

track = np.tile(track, [rep_init, 1, 1])
for r in range(rep_init, repeat):
    flg = True
    while flg:
        ic = rng.permutation(bar)
        co = np.arange(bar)
        new_track = np.array(track[-1])
        for b in range(0, bar, 2):
            if rng.uniform()>=prob_co: continue
            flg = False
            co[ic[b:b+2]] = co[ic[b:b+2]][::-1]
            new_track[ic[b:b+2]] = crossover(new_track[ic[b:b+2]])
        mu = np.zeros(bar, dtype=int)
        for b in range(bar):
            if rng.uniform()>=prob_mu: continue
            flg = False
            mu[b] = 1
            new_track[b] = mutation(new_track[b])
        if not flg: print(co, mu)
    track = np.vstack([track, new_track[np.newaxis]])
track_fl = np.reshape(track, [-1, bar*reso])

sz0 = 0.1
cmap = plt.cm.jet
plt.figure(figsize=[sz0*bar*reso, sz0*len(track_fl)])
plt.imshow(track_fl, cmap=cmap)
plt.show()


# In[4]:


rate = 44100
bpm = 150
fc = 440*2**(3/12)

def transform(notes, track_fl):
    repeat, bar, reso, div = notes.shape[:4]
    notes_fl = np.reshape(notes, [repeat, bar*reso, div]+list(notes.shape[4:]))
    notes_fl = notes_fl[np.arange(repeat)[:, np.newaxis], track_fl]
    return np.reshape(notes_fl, [repeat, bar, reso, div]+list(notes.shape[4:]))

def makenoise(notes, ratio, velo_func, pan=0, rate=rate, bpm=bpm, max_length=1, rrate=unit):
    print(notes.shape)
    repeat, bar, reso, div = notes.shape
    beat = reso/rrate
    r0 = rate*60/bpm
    n = int(r0*repeat*bar*beat)
    print(n)
    wave = np.zeros(n)
    for r in range(repeat):
        ii0 = r*bar*beat
        for b in range(bar):
            ii1 = ii0+b*beat
            for t in range(reso):
                ii2 = ii1+t/rrate
                for d in range(div):
                    if notes[r, b, t, d]==0: continue
                    ii3 = ii2++d/div/rrate
                    i0 = int(r0*(ii3))
                    i1 = int(r0*(ii3+max_length))
                    ir = np.minimum(i1-i0, n-i0)
                    velo = velo_func(np.arange(ir)/rate)
                    wave[i0:i1] = velo*np.repeat(rng.uniform(-1, 1, size=(ir-1)//ratio+1), ratio)[:ir]
    wave = 0.5*np.array([[1-pan], [1+pan]])*wave
    return wave

def sine(t, f):
    return np.sin(2*np.pi*f*t)
def triangle(r, eps=1e-8):
    def wave(t, f):
        seq = (f*t)%1
        coef = 2*((seq<r/2)+(1-r/2<seq))/np.maximum(r, eps)-2*(r/2<seq)*(seq<1-r/2)/np.maximum(1-r, eps)
        intr = -2*(1-r/2<seq)/np.maximum(r, eps)+(r/2<seq)*(seq<1-r/2)/np.maximum(1-r, eps)
        w = coef*seq+intr
        return w
    return wave
def square(r):
    def wave(t, f):
        seq = (f*t)%1
        w = 2*(seq<r)-1
        return w
    return wave
    
def makemelody(notes, f0, div, wave_func, velo_func, pan=0, 
               temp_func=lambda n: np.power(2, n/12), rate=rate, bpm=bpm, rrate=unit):
    print(notes.shape)
    repeat, bar, reso, div, _ = notes.shape
    beat = reso/rrate
    r0 = rate*60/bpm
    n = int(r0*repeat*bar*beat)
    print(n)
    wave = np.zeros(n)
    for r in range(repeat):
        ii0 = r*bar*beat
        for b in range(bar):
            ii1 = ii0+b*beat
            for t in range(reso):
                ii2 = ii1+t/rrate
                for d in range(div):
                    if notes[r, b, t, d, 1]==0: continue
                    tone, l = notes[r, b, t, d]
                    ii3 = ii2++d/div/rrate
                    i0 = int(r0*(ii3))
                    i1 = int(r0*(ii3+l))
                    ir = np.minimum(i1-i0, n-i0)
                    velo = velo_func(np.arange(ir)/rate)
                    wave[i0:i1] = velo*wave_func(np.arange(ir)/rate, f0*temp_func(tone))
    wave = 0.5*np.array([[1-pan], [1+pan]])*wave
    return wave


# In[5]:


octave = 0
div = 1
r = 0.5
wave_func = triangle(r)
pan = 1
vmax = 0.5
a = 0.5
velo_func = lambda t: vmax*np.sin(0.5*np.pi*np.minimum(t/a, 1))
notes = np.vstack([
    [[[[[ 0, 0]]]*reso]*bar]*4, 
    *[[
        [[[[11, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[10, 2]]]+[[[ 0, 0]]]*(reso//2-1), 
         [[[ 4, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[ 3, 2]]]+[[[ 0, 0]]]*(reso//2-1)], 
        [[[[-1, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[-2, 2]]]+[[[ 0, 0]]]*(reso//2-1)]*2, 
        [[[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-6, 1/2]], [[ 1, 1/2]], [[ 6, 1]], [[ 0, 0]], [[-5, 1/2]], [[ 2, 1/2]], [[ 7, 1]], [[ 0, 0]]], 
         [[[-6, 1/2]], [[ 1, 1/2]], [[ 6, 1]], [[ 0, 0]], [[-5, 1/2]], [[ 2, 1/2]], [[ 7, 1]], [[ 0, 0]]]], 
        [[[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-6, 1/2]], [[ 1, 1/2]], [[ 6, 1]], [[ 0, 0]], [[-5, 1/2]], [[ 2, 1/2]], [[ 7, 1]], [[ 0, 0]]], 
         [[[-4, 1/2]], [[ 3, 1/2]], [[ 8, 1]], [[ 0, 0]], [[-3, 1/2]], [[ 4, 1/2]], [[ 9, 1]], [[ 0, 0]]]], 
    ]]*2, 
])
cin = len(notes)
notes = np.vstack([
    notes, 
    [
        [[[[11, 2*beat]]]+[[[ 0, 0]]]*(reso-1)]+[[[[ 0, 0]]]*reso]*(bar-1), 
        [[[[ 0, 0]]]*reso]*bar, 
        [[[[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 6, 1/2]], [[ 2, 1/2]], [[ 1, 1/2]], [[ 2, 1/2]], [[-1, 1/2]]], 
         [[[ 2, 1]], [[ 0, 0]], [[14, 1]], [[ 0, 0]], [[13, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]]], 
         [[[11, 1/2]], [[13, 1/2]], [[14, 1/2]], [[18, 1/2]], [[14, 1/2]], [[13, 1/2]], [[ 9, 1/2]], [[ 1, 1/2]]], 
         [[[ 2, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]], [[ 6, 1/2]], [[ 9, 1/2]]]], 
        [[[[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[13, 1/2]], [[14, 1/2]], [[13, 1/2]], [[16, 1/2]], [[13, 1/2]]], 
         [[[14, 1/2]], [[13, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[ 2, 1]], [[ 0, 0]], [[ 1, 1]], [[ 0, 0]]], 
         [[[-1, 1/2]], [[ 1, 1/2]], [[ 2, 1/2]], [[ 6, 1/2]], [[ 2, 1/2]], [[ 6, 1/2]], [[13, 1/2]], [[14, 1/2]]], 
         [[[11, 1/2]], [[ 9, 1/2]], [[ 6, 1/2]], [[ 9, 1/2]], [[11, 2]], [[ 0, 0]], [[ 0, 0]], [[ 0, 0]]]], 
    ], 
    *[[
        [[[[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 6, 1/2]], [[ 2, 1/2]], [[ 1, 1/2]], [[ 2, 1/2]], [[-1, 1/2]]], 
         [[[ 2, 1]], [[ 0, 0]], [[14, 1]], [[ 0, 0]], [[13, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]]], 
         [[[11, 1/2]], [[13, 1/2]], [[14, 1/2]], [[18, 1/2]], [[14, 1/2]], [[13, 1/2]], [[ 9, 1/2]], [[ 1, 1/2]]], 
         [[[ 2, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]], [[ 6, 1/2]], [[ 9, 1/2]]]], 
        [[[[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[13, 1/2]], [[14, 1/2]], [[13, 1/2]], [[16, 1/2]], [[13, 1/2]]], 
         [[[14, 1/2]], [[13, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[ 2, 1]], [[ 0, 0]], [[ 1, 1]], [[ 0, 0]]], 
         [[[-1, 1/2]], [[ 1, 1/2]], [[ 2, 1/2]], [[ 6, 1/2]], [[ 2, 1/2]], [[ 6, 1/2]], [[13, 1/2]], [[14, 1/2]]], 
         [[[11, 1/2]], [[ 9, 1/2]], [[ 6, 1/2]], [[ 9, 1/2]], [[11, 2]], [[ 0, 0]], [[ 0, 0]], [[ 0, 0]]]], 
    ]]*2, 
])
cout = len(notes)
notes = np.vstack([
    notes, 
    [
        [[[[11, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[10, 2]]]+[[[ 0, 0]]]*(reso//2-1), 
         [[[ 4, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[ 3, 2]]]+[[[ 0, 0]]]*(reso//2-1)], 
        [[[[-1, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[-2, 2]]]+[[[ 0, 0]]]*(reso//2-1)]*2, 
        [[[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-6, 1/2]], [[ 1, 1/2]], [[ 6, 1]], [[ 0, 0]], [[-5, 1/2]], [[ 2, 1/2]], [[ 7, 1]], [[ 0, 0]]], 
         [[[-6, 1/2]], [[ 1, 1/2]], [[ 6, 1]], [[ 0, 0]], [[-5, 1/2]], [[ 2, 1/2]], [[ 7, 1]], [[ 0, 0]]]], 
        [[[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-8, 1/2]], [[-1, 1/2]], [[ 4, 1]], [[ 0, 0]], [[-7, 1/2]], [[ 0, 1/2]], [[ 5, 1]], [[ 0, 0]]], 
         [[[-6, 1/2]], [[ 1, 1/2]], [[ 6, 1]], [[ 0, 0]], [[-5, 1/2]], [[ 2, 1/2]], [[ 7, 1]], [[ 0, 0]]], 
         [[[-4, 1/2]], [[ 3, 1/2]], [[ 8, 1]], [[ 0, 0]], [[-3, 1/2]], [[ 4, 1/2]], [[ 9, 1]], [[ 0, 0]]]], 
    ], 
    [
        [[[[11, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[10, 2]]]+[[[ 0, 0]]]*(reso//2-1), 
         [[[ 4, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[ 3, 2]]]+[[[ 0, 0]]]*(reso//2-1)], 
        [[[[-1, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[-2, 2]]]+[[[ 0, 0]]]*(reso//2-1)]*2, 
        [[[[-1, 6]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*(reso//2)+[[[-2, 2]]]+[[[ 0, 0]]]*(reso//2-1), 
         [[[-1, 8]]]+[[[ 0, 0]]]*(reso-1), 
         [[[ 0, 0]]]*reso], 
        [[[[ 0, 0]]]*reso]*bar, 
    ], 
])
outro = 8
notes = np.vstack([
    notes, 
    [[[[[ 0, 0]]]*reso]*bar]*outro, 
])
notes[ cin:cout, ..., 0] += -1
notes[..., 0] += 12*octave
# notes = transform(notes, track_fl)
k = 10
sgm = 0.05
wave_st = np.sum([makemelody(notes, np.power(2, sgm*rng.normal())*fc, div, wave_func, velo_func, (1-2*c)*pan) for c in range(2)], axis=0)
for i in range(1, k):
    wave_st += np.sum([makemelody(notes, np.power(2, sgm*rng.normal())*fc, div, wave_func, velo_func, (1-2*c)*(1-i/k)*pan) for c in range(2)], axis=0)
wave_st /= k
repeat = int(len(wave_st[0])/rate*bpm/60/bar/beat)
print('st done.')

# wave = wave_st
# wave = wave/np.max(np.abs(wave))
# Audio(wave, rate=rate)


# In[6]:



octave = 0
div = 1
wave_func = sine
pan = 0
vmax = 0.3
sgm = 0
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]], [[14, 1/2]], [[12, 1/2]], [[14, 1/2]], [[11, 1/2]], [[ 7, 1/2]]], 
         [[[ 4, 1/2]], [[10, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[ 9, 1/2]], [[ 5, 1/2]], [[ 2, 1/2]], [[10, 1/2]]], 
         [[[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]], [[14, 1/2]], [[12, 1/2]], [[14, 1/2]], [[11, 1/2]], [[ 7, 1/2]]], 
         [[[ 4, 1/2]], [[10, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[ 9, 1/2]], [[ 4, 1/2]], [[-1, 1/2]], [[ 6, 1/2]]]], 
        [[[[ 7, 1/2]], [[ 4, 1/2]], [[ 0, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]]], 
         [[[ 0, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[-1, 1/2]], [[ 5, 1/2]], [[ 3, 1/2]], [[-2, 1/2]], [[ 6, 1/2]]], 
         [[[ 7, 1/2]], [[ 4, 1/2]], [[ 0, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]]], 
         [[[ 0, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[-1, 1/2]], [[ 5, 1/2]], [[ 3, 1/2]], [[-2, 1/2]], [[ 6, 1/2]]]]
    ]*2]*(cin//4), 
    *[[
        [[[[11, 1]], [[ 9, 1]], [[11, 1]], [[ 6, 1]], [[ 2, 1]], [[ 1, 1]], [[ 2, 1]], [[-1, 1]]], 
         [[[ 2, 2]], [[ 0, 0]], [[14, 2]], [[ 0, 0]], [[13, 2]], [[ 0, 0]], [[ 9, 2]], [[ 0, 0]]], 
         [[[11, 1]], [[13, 1]], [[14, 1]], [[18, 1]], [[14, 1]], [[13, 1]], [[ 9, 1]], [[ 1, 1]]], 
         [[[ 2, 2]], [[ 0, 0]], [[ 9, 2]], [[ 0, 0]], [[11, 2]], [[ 0, 0]], [[ 6, 1]], [[ 9, 1]]]], 
        [[[[11, 1]], [[ 9, 1]], [[11, 1]], [[13, 1]], [[14, 1]], [[13, 1]], [[16, 1]], [[13, 1]]], 
         [[[14, 1]], [[13, 1]], [[11, 1]], [[ 9, 1]], [[ 2, 2]], [[ 0, 0]], [[ 1, 2]], [[ 0, 0]]], 
         [[[-1, 1]], [[ 1, 1]], [[ 2, 1]], [[ 6, 1]], [[ 2, 1]], [[ 6, 1]], [[13, 1]], [[14, 1]]], 
         [[[11, 1]], [[ 9, 1]], [[ 6, 1]], [[ 9, 1]], [[11, 4]], [[ 0, 0]], [[ 0, 0]], [[ 0, 0]]]], 
    ]*2]*((cout-cin)//4), 
    *[[
        [[[[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]], [[14, 1/2]], [[12, 1/2]], [[14, 1/2]], [[11, 1/2]], [[ 7, 1/2]]], 
         [[[ 4, 1/2]], [[10, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[ 9, 1/2]], [[ 5, 1/2]], [[ 2, 1/2]], [[10, 1/2]]], 
         [[[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]], [[14, 1/2]], [[12, 1/2]], [[14, 1/2]], [[11, 1/2]], [[ 7, 1/2]]], 
         [[[ 4, 1/2]], [[10, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[ 9, 1/2]], [[ 4, 1/2]], [[-1, 1/2]], [[ 6, 1/2]]]], 
        [[[[ 7, 1/2]], [[ 4, 1/2]], [[ 0, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]]], 
         [[[ 0, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[-1, 1/2]], [[ 5, 1/2]], [[ 3, 1/2]], [[-2, 1/2]], [[ 6, 1/2]]], 
         [[[ 7, 1/2]], [[ 4, 1/2]], [[ 0, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]]], 
         [[[ 0, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[-1, 1/2]], [[ 5, 1/2]], [[ 3, 1/2]], [[-2, 1/2]], [[ 6, 1/2]]]]
    ]*2]*((repeat-cout-outro)//4), 
    [[[[[ 0, 0]]]*reso]*bar]*outro, 
])
notes[cin:cout, ..., 0] += -1
notes[..., 0] += 12*octave
# notes = transform(notes, track_fl[:repeat-cout+cin])
wave_ml = makemelody(notes, fc, div, wave_func, velo_func, pan)
print('ml done.')

octave = 0
div = 1
r = 0
wave_func = triangle(r)
pan = -0.7
vmax = 0.15
sgm = 0
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]], [[14, 1/2]], [[12, 1/2]], [[14, 1/2]], [[11, 1/2]], [[ 7, 1/2]]], 
         [[[ 4, 1/2]], [[10, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[ 9, 1/2]], [[ 5, 1/2]], [[ 2, 1/2]], [[10, 1/2]]], 
         [[[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]], [[14, 1/2]], [[12, 1/2]], [[14, 1/2]], [[11, 1/2]], [[ 7, 1/2]]], 
         [[[ 4, 1/2]], [[10, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[ 9, 1/2]], [[ 4, 1/2]], [[-1, 1/2]], [[ 6, 1/2]]]], 
        [[[[ 7, 1/2]], [[ 4, 1/2]], [[ 0, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]]], 
         [[[ 0, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[-1, 1/2]], [[ 5, 1/2]], [[ 3, 1/2]], [[-2, 1/2]], [[ 6, 1/2]]], 
         [[[ 7, 1/2]], [[ 4, 1/2]], [[ 0, 1/2]], [[11, 1/2]], [[ 9, 1/2]], [[11, 1/2]], [[ 7, 1/2]], [[ 4, 1/2]]], 
         [[[ 0, 1/2]], [[ 6, 1/2]], [[ 3, 1/2]], [[-1, 1/2]], [[ 5, 1/2]], [[ 3, 1/2]], [[-2, 1/2]], [[ 6, 1/2]]]]
    ]*2]*((repeat-cout+cin)//4), 
])
notes[..., 0] += 12*octave
notes = transform(notes, track_fl[:repeat-cout+cin])
notes = np.vstack([
    notes[:cin], 
    [[[[[ 0, 0]]]*reso]*bar]*(cout-cin), 
    notes[cin:], 
])
wave_hr = makemelody(notes, fc, div, wave_func, velo_func, pan)
print('hr done.')

octave = -2
div = 1
r = 1
wave_func = triangle(r)
pan = 0.7
vmax = 0.25
sgm = 5
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[[ 4, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]]], 
         [[[ 4, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 3, 1/2]]], 
         [[[ 4, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]]], 
         [[[ 4, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 3, 1/2]]]], 
        [[[[ 0, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]]], 
         [[[ 0, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-1, 1/2]]], 
         [[[ 0, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]]], 
         [[[ 0, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-1, 1/2]]]], 
    ]*2]*((repeat-cout+cin)//4), 
])
notes[..., 0] += 12*octave
notes = transform(notes, track_fl[:repeat-cout+cin])
notes = np.vstack([
    notes[:cin], 
    [[[[[ 0, 0]]]*reso]*bar]*(cout-cin), 
    notes[cin:], 
])
wave_md = makemelody(notes, fc, div, wave_func, velo_func, pan)
print('md done.')

octave = -4
div = 1
r = 0.5
wave_func = square(r)
pan = 0
vmax = 0.5
sgm = 5
velo_func = lambda t: vmax*np.exp(-sgm*t)

notes = np.vstack([
    *[[
        [[[[ 4, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]]], 
         [[[ 4, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 3, 1/2]]], 
         [[[ 4, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]]], 
         [[[ 4, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 3, 1/2]]]], 
        [[[[ 0, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]]], 
         [[[ 0, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-1, 1/2]]], 
         [[[ 0, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]]], 
         [[[ 0, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-1, 1/2]]]], 
    ]*2]*(cin//4), 
    [
        *[
            [[[[ 7, 3]]]+[[[ 0, 0]]]*(unit*(beat-1)-1)+[[[ 9, 1]], [[ 0, 0]]], 
             [[[11, 3]]]+[[[ 0, 0]]]*(unit*(beat-1)-1)+[[[ 9, 1]], [[ 0, 0]]], 
             [[[ 7, 3]]]+[[[ 0, 0]]]*(unit*(beat-1)-1)+[[[ 9, 1]], [[ 0, 0]]], 
             [[[11, 3]]]+[[[ 0, 0]]]*(unit*(beat-1)-1)+[[[11, 1]], [[ 0, 0]]]], 
        ]*2, 
        *[
            [[[[ 7, 1]], [[ 0, 0]], [[ 7, 1]], [[ 0, 0]], [[ 7, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]]], 
             [[[11, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]]], 
             [[[ 7, 1]], [[ 0, 0]], [[ 7, 1]], [[ 0, 0]], [[ 7, 1]], [[ 0, 0]], [[ 9, 1]], [[ 0, 0]]], 
             [[[11, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]], [[11, 1]], [[ 0, 0]]]], 
        ]*(cout-cin-2), 
    ], 
    *[[
        [[[[ 4, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]]], 
         [[[ 4, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 3, 1/2]]], 
         [[[ 4, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 5, 1/2]], [[ 4, 1/2]], [[ 4, 1/2]]], 
         [[[ 4, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 3, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 3, 1/2]]]], 
        [[[[ 0, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]]], 
         [[[ 0, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-1, 1/2]]], 
         [[[ 0, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 2, 1/2]], [[ 0, 1/2]], [[ 0, 1/2]]], 
         [[[ 0, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-1, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-2, 1/2]], [[-1, 1/2]]]], 
    ]*2]*((repeat-cout)//4), 
])
notes[cin:cout, ..., 0] += -1
notes[..., 0] += 12*octave
# notes = transform(notes, track_fl[:repeat-cout+cin])
wave_ba = makemelody(notes, fc, div, wave_func, velo_func, pan)
print('ba done.')

wave_pt = wave_st+wave_ml+wave_hr+wave_md+wave_ba
# wave = wave_pt
# wave = wave/np.max(np.abs(wave))
# Audio(wave, rate=rate)


# In[7]:


ratio = 64
pan = 0
vmax = 1
sgm = 15
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]], 
         [[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]], 
         [[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], 
         [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]]], 
    ]]*(repeat-cout+cin), 
])
notes = transform(notes, track_fl[:repeat-cout+cin])
notes = np.vstack([
    notes[:cin], 
    *[[
        [[[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]], 
         [[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]], 
         [[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]], 
         [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]]], 
    ]]*(cout-cin), 
    notes[cin:], 
])
notes[cout-1, -1, -2:] = 0
wave_bd = makenoise(notes, ratio, velo_func, pan)
print('bd done.')

ratio = 16
pan = -0.2
vmax = 0.8
sgm = 25
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[0, 0], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1]], 
         [[0, 0], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1]], 
         [[0, 0], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [1, 0]], 
         [[0, 1], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [1, 0]]], 
    ]]*(repeat-cout+cin), 
])
notes = transform(notes, track_fl[:repeat-cout+cin])
notes = np.vstack([
    notes[:cin], 
    [[[[0, 0]]*reso]*bar]*2, 
    [
        [[[0, 0], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1]], 
         [[0, 0], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1]], 
         [[0, 0], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [1, 0]], 
         [[0, 1], [0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [1, 0]]], 
    ]*(cout-cin-2), 
    notes[cin:], 
])
notes[cout-1, -1, -2:] = 0
wave_sd = makenoise(notes, ratio, velo_func, pan)
print('sd done.')

ratio = 2
pan = 0.3
vmax = 0.6
sgm = 50
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[1], [1], [1], [1], [1], [1], [1], [1]], 
         [[1], [1], [1], [1], [1], [1], [1], [1]], 
         [[1], [1], [1], [1], [1], [1], [1], [1]], 
         [[1], [1], [1], [1], [1], [0], [1], [1]]], 
    ]]*(repeat-cout+cin), 
])
notes = transform(notes, track_fl[:repeat-cout+cin])
notes = np.vstack([
    notes[:cin], 
    [[[[0]]*reso]*bar]*4, 
    [
        [[[1], [1], [1], [1], [1], [1], [1], [1]], 
         [[1], [1], [1], [1], [1], [1], [1], [1]], 
         [[1], [1], [1], [1], [1], [1], [1], [1]], 
         [[1], [1], [1], [1], [1], [0], [1], [1]]], 
    ]*(cout-cin-4), 
    notes[cin:], 
])
notes[cout-1, -1, -2:] = 0
wave_hh = makenoise(notes, ratio, velo_func, pan)
print('hh done.')

ratio = 2
pan = 0.3
vmax = 0.6
sgm = 10
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [1], [0], [0]]], 
    ]]*(repeat-cout+cin), 
])
notes = transform(notes, track_fl[:repeat-cout+cin])
notes = np.vstack([
    notes[:cin], 
    [[[[0]]*reso]*bar]*4, 
    [
        [[[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [1], [0], [0]]], 
    ]*(cout-cin-4), 
    notes[cin:], 
])
notes[cout-1, -1, -2:] = 0
wave_oh = makenoise(notes, ratio, velo_func, pan, max_length=1/unit)
print('oh done.')

ratio = 4
pan = -0.5
vmax = 0.4
sgm = 2
velo_func = lambda t: vmax*np.exp(-sgm*t)
notes = np.vstack([
    *[[
        [[[1], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]], 
         [[0], [0], [0], [0], [0], [0], [0], [0]]], 
    ]]*(repeat-outro+1), 
    [[[[0]]*reso]*bar]*(outro-1), 
])
# notes = transform(notes, track_fl)
wave_cc = makenoise(notes, ratio, velo_func, pan, max_length=2*beat)
print('cc done.')

wave_dr = wave_bd+wave_sd+wave_hh+wave_oh+wave_cc
# wave = wave_dr
# wave = wave/np.max(np.abs(wave))
# Audio(wave, rate=rate)


# In[8]:


wave = wave_pt+wave_dr
fo = np.ones(wave.shape[1])
r = 2
# alp = 0.5
# fo[int(rate*60/bpm*(repeat-r)*bar*beat):] = np.exp(-alp*np.linspace(0, int(60/bpm*r*bar*beat), int(rate*60/bpm*r*bar*beat), False))
fo[int(rate*60/bpm*(repeat-r)*bar*beat):] = np.linspace(1, 0, int(rate*60/bpm*r*bar*beat))
wave = fo*wave
wave = wave/np.max(np.abs(wave))
Audio(wave, rate=rate)


# In[ ]:




