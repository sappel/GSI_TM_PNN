#autogenerated on 2020/08/04 16:10:25 
from ocelot import *
raddeg =1/57.2958
ll = 150*raddeg/5
alpha = 15*raddeg
pfr = 7.3*raddeg
fi = 0.7
none = 0
ksf=0
ksd=0
rb = SBend(l=ll,angle=alpha,e1=pfr,e2=pfr,fint=fi)
qs1f = Quadrupole(l=1.04/5,k1=0.30989596)
qs2d = Quadrupole(l=1.04/5,k1=-0.49964116)
qs3t = Quadrupole(l=0.4804/5,k1=0.62221964)
sf = Sextupole(l=0.0001,k2=0.0)
sd = Sextupole(l=0.0001,k2=-0.0)
bpm = Monitor()
cor = Hcor(l=0,angle=0)
sis18cell = Marker()
ring=(Drift(l=0.6450004680544859),
rb, rb, rb, rb, rb,
Drift(l=0.9700009361089719),
rb, rb, rb, rb, rb,
Drift(l=6.503962172054486),sf,
Drift(l=0.3349499999999992),bpm,
cor,
qs1f, qs1f, qs1f, qs1f, qs1f,
Drift(l=0.5999999999999996),
qs2d, qs2d, qs2d, qs2d, qs2d, 
Drift(l=0.35494999999999877),sd,
Drift(l=0.3547499999999957),
qs3t, qs3t, qs3t, qs3t, qs3t, 
Drift(l=0.49980053999999896),bpm,
Marker() )

#for i in range(len(ring)): # to avoid new variable name in for
#    id_list = [ k for k,v in locals().items() if v == ring[i]]
#    ring[i].cs_id = id_list