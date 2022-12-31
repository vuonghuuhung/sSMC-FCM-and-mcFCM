import numpy as np

def theLeft(mU, alpha):
    return mU*(alpha**(mU-1))

def SolvingFunc(uik, mL, alpha):
    mU = max(-1/np.log(alpha),mL)
    temp = (1-alpha)/(1/uik - 1)
    theRight = mL*(temp**(mL-1))
    while theLeft(mU,alpha) > theRight:
        # print(theLeft(mU,alpha),theRight)
        mU+=0.05

    # high,low = mU,mU-0.5
    # mU= (high+low)/2
    # i=0
    # while abs(theLeft(mU,alpha) - theRight)>0.0001 and i<10:
    #     i+=1
    #     # print(mU,uik,theLeft(mU,alpha),theRight)
    #     if theLeft(mU,alpha) > theRight:
    #         low = mU
    #     else: high = mU
    #     mU= (low+high)/2
    return mU

for i in range(1,10):
    print(np.round(SolvingFunc(i/10,2,0.6),2))

