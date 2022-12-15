#!/usr/bin/env python
# coding: utf-8

# In[124]:


global k

k=[]
k.append('01000010100010100010111110011000')
k.append('01110001001101110100010010010001')
k.append('10110101110000001111101111001111')
k.append('11101001101101011101101110100101')
k.append('00111001010101101100001001011011')
k.append('01011001111100010001000111110001')
k.append('10010010001111111000001010100100')
k.append('10101011000111000101111011010101')
k.append('11011000000001111010101010011000')
k.append('00010010100000110101101100000001')
k.append('00100100001100011000010110111110')
k.append('01010101000011000111110111000011')
k.append('01110010101111100101110101110100')
k.append('10000000110111101011000111111110')
k.append('10011011110111000000011010100111')
k.append('11000001100110111111000101110100')
k.append('11100100100110110110100111000001')
k.append('11101111101111100100011110000110')
k.append('00001111110000011001110111000110')
k.append('00100100000011001010000111001100')
k.append('00101101111010010010110001101111')
k.append('01001010011101001000010010101010')
k.append('01011100101100001010100111011100')
k.append('01110110111110011000100011011010')
k.append('10011000001111100101000101010010')
k.append('10101000001100011100011001101101')
k.append('10110000000000110010011111001000')
k.append('10111111010110010111111111000111')
k.append('11000110111000000000101111110011')
k.append('11010101101001111001000101000111')
k.append('00000110110010100110001101010001')
k.append('00010100001010010010100101100111')
k.append('00100111101101110000101010000101')
k.append('00101110000110110010000100111000')
k.append('01001101001011000110110111111100')
k.append('01010011001110000000110100010011')
k.append('01100101000010100111001101010100')
k.append('01110110011010100000101010111011')
k.append('10000001110000101100100100101110')
k.append('10010010011100100010110010000101')
k.append('10100010101111111110100010100001')
k.append('10101000000110100110011001001011')
k.append('11000010010010111000101101110000')
k.append('11000111011011000101000110100011')
k.append('11010001100100101110100000011001')
k.append('11010110100110010000011000100100')
k.append('11110100000011100011010110000101')
k.append('00010000011010101010000001110000')
k.append('00011001101001001100000100010110')
k.append('00011110001101110110110000001000')
k.append('00100111010010000111011101001100')
k.append('00110100101100001011110010110101')
k.append('00111001000111000000110010110011')
k.append('01001110110110001010101001001010')
k.append('01011011100111001100101001001111')
k.append('01101000001011100110111111110011')
k.append('01110100100011111000001011101110')
k.append('01111000101001010110001101101111')
k.append('10000100110010000111100000010100')
k.append('10001100110001110000001000001000')
k.append('10010000101111101111111111111010')
k.append('10100100010100000110110011101011')
k.append('10111110111110011010001111110111')
k.append('11000110011100010111100011110010')

   
A = '01101010000010011110011001100111'
B = '10111011011001111010111010000101'
C = '00111100011011101111001101110010'
D = '10100101010011111111010100111010'
E = '01010001000011100101001001111111'
F = '10011011000001010110100010001100'
G = '00011111100000111101100110101011'
H = '01011011111000001100110100011001'

global A
global B
global C
global D
global E
global F
global G
global H




# In[125]:


def RotR(X,m):
    Y=''
    for i in range(len(X)):
        Y += X[(i-m)%len(X)]
    return Y


# In[126]:


def ShR(X,m):
    Y=''
    for i in range(len(X)):
        if(i<m):
            Y += '0'
        else:
            Y += X[(i-m)%len(X)]
    return Y


# In[127]:


def s0(X):
    lista=''
    a = RotR(X,7)
    b = RotR(X,18)
    c = ShR(X,3)
    for i in range(32):
        lista += str((int(a[i]) + int(b[i]) + int(c[i]))%2)
    return lista


# In[128]:


def s1(X):
    lista=''
    a = RotR(X,17)
    b = RotR(X,19)
    c = ShR(X,10)
    for i in range(32):
        lista += str((int(a[i]) + int(b[i]) + int(c[i]))%2)
    return lista


# In[129]:


def S0(X):
    lista=''
    a = RotR(X,2)
    b = RotR(X,13)
    c = RotR(X,22)
    for i in range(32):
        lista += str((int(a[i]) + int(b[i]) + int(c[i]))%2)
    return lista


# In[130]:


def S1(X):
    lista=''
    a = RotR(X,6)
    b = RotR(X,11)
    c = RotR(X,25)
    for i in range(32):
        lista += str((int(a[i]) + int(b[i]) + int(c[i]))%2)
    return lista


# In[131]:


def sum1(X,Y):
    z=''
    c=0
    for i in range(31,-1,-1):
        z += str((c + int(X[i])+int(Y[i]))%2)
        c = (int(X[i])*int(Y[i]) + c*int(X[i]) + c*int(Y[i]))%2    
    return z[::-1] #sale como z32...z1


# In[132]:


def majority(x,y,z):
    return ((int(x)+int(y))*(int(y)+int(z)) + int(y))%2


# In[133]:


def choice(x,y,z):
    return (int(x)*(int(y)+int(z)) + int(z))%2


# In[134]:


def padding(M):
        
    o=''

    l = ''.join("{0:08b}".format(int(int.from_bytes(bytes(bytearray(p.encode('utf-8'))), byteorder='big')), 'b') for p in M)

    o = ''.join("{0:064b}".format(len(l), 'b'))  #Original message length, represented in binary, 64 bits
    
    l+='10000000'                                #Add one 1 and seven 0 to the original message
    
    while((len(l)+64)%512 != 0):
        l+='0'                                   #Padding
    l+= o                                        #Add at the very end the 64 bits that represents the length of
                                                 #the original message
    
    return l


# In[135]:


def iteration(a,b,c,d,e,f,g,h,l):    
    wit = ''
    W=[]
    l1=[]
    l2=[]
    
    Cho=[]
    Temp11=[]
    Temp12=[]
    Temp13=[]
    Temp1out=[]
    
    Maj=[]
    Temp2out=[]
    
    elist=[]
     
    for t in range(16):                          #First 16 variables W
        W.append(l[32*t:32*(t+1)]) 
        wit += l[32*t:32*(t+1)][::-1]
        
    for i in range(16,64):
        t1 = sum1(W[i-16],W[i-7])
        t2 = sum1(t1,s1(W[i-2]))
        t3 = sum1(t2,s0(W[i-15]))
        W.append(t3)                                 #All variables W
        l1.append(t1)
        l2.append(t2)
    
    for t in range(16,64):
        wit += W[t][::-1]   #REVERSE STRING, we need to do this in order to call the rightest bit as the first
        
    for t in range(16,64):
        wit += l1[t-16][::-1]   
        
    for t in range(16,64):
        wit += l2[t-16][::-1]       
            
    wit += d[::-1]+c[::-1]+b[::-1]+a[::-1] #In order to get a patron, we add this variables to the witness now. 
    
    a0 = a
    b0 = b
    c0 = c
    d0 = d
    e0 = e
    f0 = f
    g0 = g
    h0 = h
    
    for j in range(64):
    #___________________________________________________________________________
    
        Ch=''                                                #Temp1
        for i in range(32):
            Ch += str(choice(e0[i],f0[i],g0[i]))
        Cho.append(Ch)
    
        e1 = S1(e0)
    
        t1 = sum1(h0,e1)
        Temp11.append(t1)
        
        t2 = sum1(t1,Ch)
        Temp12.append(t2)
        
        t3 = sum1(t2,k[j])
        Temp13.append(t3)   
        
        temp1 = sum1(t3,W[j])
        Temp1out.append(temp1)
  #_________________________________________________________________________ 
        
        
        a1 = S0(a0)                                          #Temp2
        m = ''
    
        for i in range(32):
            m += str(majority(a0[i],b0[i],c0[i]))
        Maj.append(m)
            
        temp2 = sum1(a1,m)
        Temp2out.append(temp2)
  #_______________________________________________________________________   

        h0 = g0
        g0 = f0
        f0 = e0
        e0 = sum1(d0,temp1)
        
        elist.append(e0)
        
        d0 = c0
        c0 = b0
        b0 = a0
        a0 = sum1(temp1,temp2)
        
        wit += a0[::-1]
    
    wit += h[::-1]+g[::-1]+f[::-1]+e[::-1]
    
    for t in range(64):
        wit += elist[t][::-1]
        
    for t in range(64):
        wit += Cho[t][::-1]

    for t in range(64):
        wit += Temp11[t][::-1]
        
    for t in range(64):
        wit += Temp12[t][::-1]
        
    for t in range(64):
        wit += Temp13[t][::-1]
        
    for t in range(64):
        wit += Temp1out[t][::-1]
        
    for t in range(64):
        wit += Maj[t][::-1] 
        
    for t in range(64):
        wit += Temp2out[t][::-1]
        
    for t in range(64):
        wit += k[t][::-1]
        
    a = sum1(a0,a)
    b = sum1(b0,b)
    c = sum1(c0,c)
    d = sum1(d0,d)
    e = sum1(e0,e)
    f = sum1(f0,f)
    g = sum1(g0,g)
    h = sum1(h0,h)
    
    wit += a[::-1]+b[::-1]+c[::-1]+d[::-1]+e[::-1]+f[::-1]+g[::-1]+h[::-1]
    
    wit = '1' + wit
    
    return a,b,c,d,e,f,g,h,wit
    
    
    


# In[136]:


def SHA256(M):
    
    l = padding(M)
    
    v = iteration(A,B,C,D,E,F,G,H,l)

    while(len(l)>512):
        l = l[512:]
        v = iteration(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],l)
        
    a1 = "{0:0>8x}".format(int(v[0], 2))
    b1 = "{0:0>8x}".format(int(v[1], 2))
    c1 = "{0:0>8x}".format(int(v[2], 2))
    d1 = "{0:0>8x}".format(int(v[3], 2))
    e1 = "{0:0>8x}".format(int(v[4], 2))
    f1 = "{0:0>8x}".format(int(v[5], 2))
    g1 = "{0:0>8x}".format(int(v[6], 2))
    h1 = "{0:0>8x}".format(int(v[7], 2))
    
    result = a1+b1+c1+d1+e1+f1+g1+h1
    
    print(result)
    
    return v[8]


# In[137]:


#Now, the witness order follows:

#1,w_1¹,...,w³²_1,w¹_2,...,w³²_16,w¹_17,...,w³²_64,t1¹_1,...,t1¹_2,...,t1³²_48,t2¹_1...,t2³²_48,

#(now each variable is supposed to have 32 bits)

#d0,c0,b0,a0,a1,a2,....,a64,h0,g0,f0,e0,e1,e2,...,e64,Ch_1,...,Ch_64,sum1_1,...,sum1_64,sum2_1,...,sum3_64
#Temp1_1,....,Temp1_64,Maj_1,...,Maj_64,Temp2_1,...,Temp2_64,k_1,...,k_64,H_1,...,H_8


# In[138]:


def Choice_R1CS(m1,m2,m3,row,e,f,g,Ch): #Put the exact witness position for e,f,g,Ch 
    
    for i in range(32):
    
        m1[row+i][e-1+i] = 1
    
        m2[row+i][f-1+i] = 1
        m2[row+i][g-1+i] = 1
    
        m3[row+i][Ch-1+i] = 1
        m3[row+i][g-1+i] = 1
    
    return m1,m2,m3


# In[139]:


def Majority_R1CS(m1,m2,m3,row,a,b,c,Maj): #Put the exact witness position for a,b,c,Maj
            
    for i in range(32):
        
        m1[row+i][a-1+i] = 1
        m1[row+i][b-1+i] = 1
        
        m2[row+i][c-1+i] = 1
        m2[row+i][b-1+i] = 1
        
        m3[row+i][b-1+i] = 1
        m3[row+i][Maj-1+i] = 1
        
    return m1,m2,m3
            


# In[140]:


def Sumod_2_32_R1CS(m1,m2,m3,row,sum1,sum2,total): #Put the exact witness position for sum1,sum2,total
    
    #scheme for sum modulo 2^32                         [a+b=z]
    
         # z1 = a1 + b1,  ||     0*0 +a1+b1+z1 = 0
         # z2 = a1*b1 + a2 + b2
         # zi = (zi-1 + bi-1)*(ai-1 + bi-1) + ai + bi + ai-1
    
    m3[row][sum1-1] = 1
    m3[row][sum2-1] = 1
    m3[row][total-1] = 1
#______________________________
    
    m1[row+1][sum1-1] = 1
    
    m2[row+1][sum2-1] = 1
    
    m3[row+1][sum1] = 1
    m3[row+1][sum2] = 1
    m3[row+1][total] = 1
    
    
    for i in range(2,32):

        m1[row+i][total-2+i] = 1 #total^(i-1)
        m1[row+i][sum2-2+i] = 1 #sum2^(i-1)
        
        m2[row+i][sum1-2+i] = 1 #sum1^(i-1)
        m2[row+i][sum2-2+i] = 1 #sum2^(i-1)
        
        m3[row+i][sum1-1+i] = 1 #sum1^(i)
        m3[row+i][sum2-1+i] = 1 #sum2^(i)
        m3[row+i][sum1-2+i] = 1 #sum1^(i-1)
        m3[row+i][total-1+i] = 1 #total^(i)
        
    return m1,m2,m3        
        


# In[141]:


def Sumod_2_32_R1CS_s(m1,m2,m3,row,sum1,s,total,rot1,rot2,sh,value): 
    
    #Where s is the precise position of the first bit of the variable before rotation
    #value = {0,1} determinates whether a shift rigth rotation is required
    
    m3[row][sum1-1] = 1
    m3[row][s-1+sh] = 1
    m3[row][s-1+rot1] = 1
    m3[row][s-1+rot2] = 1
    m3[row][total-1] = 1
#______________________________
    
    m1[row+1][sum1-1] = 1
    
    m2[row+1][s-1+sh] = 1
    m2[row+1][s-1+rot1] = 1
    m2[row+1][s-1+rot2] = 1
    
    m3[row+1][sum1] = 1
    m3[row+1][s+sh] = 1
    m3[row+1][s+rot1] = 1
    m3[row+1][s+rot2] = 1
    m3[row+1][total] = 1
    
    
    for i in range(2,32):
        
        if(i<32-sh):

            m1[row+i][total-2+i] = 1 #total^(i-1)
            m1[row+i][s-1+(sh+i-1)%32] = 1 #s1^(i-1)
            m1[row+i][s-1+(rot1+i-1)%32] = 1
            m1[row+i][s-1+(rot2+i-1)%32] = 1
        
            m2[row+i][sum1-2+i] = 1 #sum1^(i-1)
            m2[row+i][s-1+(sh+i-1)%32] = 1 #s1^(i-1)
            m2[row+i][s-1+(rot1+i-1)%32] = 1
            m2[row+i][s-1+(rot2+i-1)%32] = 1
        
            m3[row+i][sum1-1+i] = 1 #sum1^(i)
            m3[row+i][s-1+(sh+i)%32] = 1 #s1^(i)
            m3[row+i][s-1+(rot1+i)%32] = 1
            m3[row+i][s-1+(rot2+i)%32] = 1
            m3[row+i][sum1-2+i] = 1 #sum1^(i-1)
            m3[row+i][total-1+i] = 1 #total^(i)
        
        elif(i==32-sh):
            
            m1[row+i][total-2+i] = 1 #total^(i-1)
            m1[row+i][s-2+sh+i] = 1 #s1^(i-1)
            m1[row+i][s-1+(rot1+i-1)%32] = 1
            m1[row+i][s-1+(rot2+i-1)%32] = 1
        
            m2[row+i][sum1-2+i] = 1 #sum1^(i-1)
            m2[row+i][s-2+sh+i] = 1 #s1^(i-1)
            m2[row+i][s-1+(rot1+i-1)%32] = 1
            m2[row+i][s-1+(rot2+i-1)%32] = 1
        
            m3[row+i][sum1-1+i] = 1 #sum1^(i)
            m3[row+i][s-1+(sh+i)%32] = 1*value #s1^(i)
            m3[row+i][s-1+(rot1+i)%32] = 1
            m3[row+i][s-1+(rot2+i)%32] = 1
            m3[row+i][sum1-2+i] = 1 #sum1^(i-1)
            m3[row+i][total-1+i] = 1 #total^(i)
            
        else:
            
            m1[row+i][total-2+i] = 1 #total^(i-1)
            m1[row+i][s-1+(sh+i-1)%32] = 1*value #s1^(i-1)
            m1[row+i][s-1+(rot1+i-1)%32] = 1
            m1[row+i][s-1+(rot2+i-1)%32] = 1
        
            m2[row+i][sum1-2+i] = 1 #sum1^(i-1)
            m2[row+i][s-1+(sh+i-1)%32] = 1*value #s1^(i-1)
            m2[row+i][s-1+(rot1+i-1)%32] = 1
            m2[row+i][s-1+(rot2+i-1)%32] = 1
        
            m3[row+i][sum1-1+i] = 1 #sum1^(i)
            m3[row+i][s-1+(sh+i)%32] = 1*value #s1^(i)
            m3[row+i][s-1+(rot1+i)%32] = 1
            m3[row+i][s-1+(rot2+i)%32] = 1
            m3[row+i][sum1-2+i] = 1 #sum1^(i-1)
            m3[row+i][total-1+i] = 1 #total^(i)
            
        
    return m1,m2,m3        
        


# In[142]:


def R1CS():
        
    import numpy as np
    
    A = np.zeros((23296, 26113)) 
    B = np.zeros((23296, 26113))
    C = np.zeros((23296, 26113))
    
    #FIRST PHASE
    
    for m in range(48):
        
        A,B,C = Sumod_2_32_R1CS(A,B,C,m*96,2+m*32,290+m*32,2050+m*32) #m1,m2,m3,row,w1,w10,t1
                
        A,B,C = Sumod_2_32_R1CS_s(A,B,C,32+m*96,2050+m*32,450+m*32,3586+m*32,17,19,10,0) #m1,m2,m3,row,t1,s1,t2 s1=w¹15
        
        A,B,C = Sumod_2_32_R1CS_s(A,B,C,64+m*96,3586+m*32,34+m*32,514+m*32,7,18,3,0) #m1,m2,m3,row,t2,s0,w17 s0=w¹_2
  
                                #m1,m2,m3,row,sum1,s,total,rot1,rot2,(sh/rot3),value
                                # if   value = 0 -> do shift rotate right
                                # else value = 1 -> do another rotate right
                                # s is the first bit position of the variable before rotation
                
    #SECOND PHASE
                
    for m in range(64):
        
        A,B,C = Choice_R1CS(A,B,C,4608+m*288,7394+m*32,7362+m*32,7330+m*32,9474+m*32) #m1,m2,m3,row,e,f,g,Ch
        A,B,C = Sumod_2_32_R1CS_s(A,B,C,4608+32+m*288,7298+m*32,7394+m*32,11522+m*32,6,11,25,1) #m1,m2,m3,row,h,S1,Sum1
        A,B,C = Sumod_2_32_R1CS(A,B,C,4608+64+m*288,11522+m*32,9474+m*32,13570+m*32) #m1,m2,m3,row,Sum1,Ch,Sum2
        A,B,C = Sumod_2_32_R1CS(A,B,C,4608+96+m*288,13570+m*32,23810+m*32,15618+m*32) #m1,m2,m3,row,Sum2,k,Sum3
        A,B,C = Sumod_2_32_R1CS(A,B,C,4608+128+m*288,15618+m*32,2+m*32,17666+m*32) #m1,m2,m3,row,Sum3,w,Temp1
        
        A,B,C = Majority_R1CS(A,B,C,4608+160+m*288,5218+m*32,5186+m*32,5154+m*32,19714+m*32)
        A,B,C = Sumod_2_32_R1CS_s(A,B,C,4608+192+m*288,19714+m*32,5218+m*32,21762+m*32,2,13,22,1) #m1,m2,m3,row,Maj1,a0,Temp2

        A,B,C = Sumod_2_32_R1CS(A,B,C,4608+224+m*288,17666+m*32,21762+m*32,5250+m*32) #m1,m2,m3,row,Temp1,Temp2,a1
        A,B,C = Sumod_2_32_R1CS(A,B,C,4608+256+m*288,17666+m*32,5122+m*32,7426+m*32) #m1,m2,m3,row,Temp1,d0,e1

                
    #THIRD PHASE || H1 = a0 + a64, H2 = b0 + a63, H3 = c0 + a62, H4 = d0 + a61
    
    for m in range(4):
        
        A,B,C = Sumod_2_32_R1CS(A,B,C,23040+m*32,7266-m*32,5218-m*32,25858+m*32) #m1,m2,m3,row,(a64,a0,H1)        
                
    #FOURTH PHASE || H5 = e0 + e64, H6 = f0 + e63, H7 = g0 + e62, H8 = h0 + e61
                
    for m in range(4):
        
        A,B,C = Sumod_2_32_R1CS(A,B,C,23168+m*32,9442-m*32,7394-m*32,25986+m*32) #m1,m2,m3,row,(e64,e0,H5) 
              
    
    return A,B,C


# In[143]:


def ultimo(M):
    
    import numpy as np
    
    matrices=R1CS()
    
    l = padding(M)
    
    v = iteration(A,B,C,D,E,F,G,H,l)
    
    wit = list(map(int,v[8]))
    print(sum((np.multiply(np.dot(matrices[0], wit),np.dot(matrices[1], wit)) + np.dot(matrices[2],wit))%2 == 0) == 23296)

    while(len(l)>512):
        
        l = l[512:]
        v = iteration(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],l)
        
        wit = list(map(int,v[8]))
        print(sum((np.multiply(np.dot(matrices[0], wit),np.dot(matrices[1], wit)) + np.dot(matrices[2],wit))%2 == 0) == 23296)
    
    a1 = "{0:0>8x}".format(int(v[0], 2))
    b1 = "{0:0>8x}".format(int(v[1], 2))
    c1 = "{0:0>8x}".format(int(v[2], 2))
    d1 = "{0:0>8x}".format(int(v[3], 2))
    e1 = "{0:0>8x}".format(int(v[4], 2))
    f1 = "{0:0>8x}".format(int(v[5], 2))
    g1 = "{0:0>8x}".format(int(v[6], 2))
    h1 = "{0:0>8x}".format(int(v[7], 2))
    
    result = a1+b1+c1+d1+e1+f1+g1+h1
    
    print(result)
    


# In[144]:


ultimo('hola esto es una prueba para comprobar si esto funciona')


# In[145]:


ultimo('hola esto es una prueba para comprobar si esto funcionaa')


# In[117]:


sum(matrices[0][:][:]==1)


# In[118]:


sum(matrices[1][:][:]==1)


# In[119]:


sum(matrices[2][:][:]==1)


# In[120]:


maxim=0
for i in range(23296):
    maxim = max(maxim,sum(matrices[0][i][:]==1))
print(maxim)


# In[121]:


maxim=0
for i in range(23296):
    maxim = max(maxim,sum(matrices[1][i][:]==1))
print(maxim)


# In[122]:


maxim=0
for i in range(23296):
    maxim = max(maxim,sum(matrices[2][i][:]==1))
print(maxim)


# In[123]:


float((55656+58152+97504)/(3*23296*26113))*100

