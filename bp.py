from math import e as e
from random import uniform as rand
#def Alpha
Alpha=0.5
#learning rate
lR=0.4
#jumlah output layer
Y=1
#Y=1
#bobot ke H
wH=[[]]
#bobot ke Y
wY=[[]]
#var wInput
wIn=[]
#var 
dH=[]
#jumlah hidden layer
layer=2
#var bobot
w=[[]]
#bias
b=[]	
#input x,y dengan 4 data
inp=[[1,1],[1,0],[0,1],[0,0]]
#inp=[[0.7778,0.8333],[0.8333,0.8889],[0.7222,0.8611],[0.9444,0.8333],[0.6111,1.000],[0.8333,0.8333]]
inp=[[1,1]]
#var output layer H dan Y
outH=[]
outY=[]
#target 
#yT=[[0],[1],[1],[0],[1],[1]]
yT=[[1],[1],[1],[0]]
#yT=[[1]]
#maksimal error
maxEr=0.1
#input nilai w secara random
def insertW():
	global wH
	wH=[]
	global wY
	wY=[]
	for _ in range(0,layer):
		tmp=[]
		for _ in range(0,len(inp[0])):
			tmp.append(rand(0,1))
		wH.append(tmp)
	for _ in range(0,Y):
		tmp=[]
		for _ in range(0,layer):
			tmp.append(rand(0,1))
		wY.append(tmp)

#input bias secara random
def insertB():
	global b
	for _ in range(0,2):
		b.append(rand(0,1))

#fungsi aktifasi
def fungsiAktifasi(val):
	return 1/(1+e**-val)

def forwardPass():
	global outY
	global outH
	outY=[]
	outH=[]
	for h in range(0,layer):
		oH=[]
		for i in range(0,len(inp)):
			tmp=0
			for j in range(0,len(inp[i])):
				tmp+=inp[i][j]*wH[h][j]
				tmp+=b[0]
			tmp=fungsiAktifasi(tmp)
			oH.append(tmp)
		outH.append(oH)
	#output layer
	for i in range(0, len(outH[0])):#6
		oY=[]
		for j in range(0,Y):  #2
			tmp=0
			for k in range(0,len(outH)): #10
				z=outH[k][i]*wY[j][k]
				tmp+=z
				#print(f'tmp : {tmp}')
				#print(f"outH :{outH[k][i]} , {wY[j][k]}= {z}")
			tmp+=b[1]
			tmp=fungsiAktifasi(tmp)
			oY.append(tmp)
		outY.append(oY)

#sampai sini

#func delta Y
def deltaY(i,j):
	x=(yT[i][j]-outY[i][j])
	y=outY[i][j]
	z=(1-outY[i][j])
	return x*y*z

#func deltan.
def delta(dY,H):
	return dY*Alpha*H
#func deltaIn H
def deltInH(a,b):
	return a*b
#func deltaH
def deltaH(a,b):
	return a*b*(1-b)

#backwardPass 
#idx= index number of data
def backwardPass(idx):
	#dY=deltaY(x,y)
	#new
	dY=[]
	outDy=0
	for i in range(0,Y):
		dY.append(deltaY(idx,i))
		outDy+=dY[i]	
	global wIn
	global dH
	inH=[]
	wIn=[]
	dH=[]
	#y to h, update weight
	for h in range(0,len(dY)):#2
		wH=[]
		inH=[]
		tmpDH=[]
		for i in range(0,layer):#10
			wH.append(delta(dY[h],outH[i][idx]))
			inH.append(deltInH(dY[h],wH[i]))
			tmpDH.append(deltaH(inH[i],outH[i][idx]))
		dH.append(tmpDH)#10x2
		b[0]=b[0]+delta(outDy,1)
	#h to i
	
	for h in range(0,len(inp[0])):#2
		tmpWIn=[]
		for i in range(0,layer):#10
			#print(f'idx : {idx}, i: {i}, h: {h}')
			#tmpWIn.append(delta(dH[idx][i],inp[idx][h]))
			tmpWIn.append(delta(dH[0][i],inp[idx][h]))
		wIn.append(tmpWIn)#10x2
		b[1]=b[1]+delta(outDy,1) #out range in here 
	updateWeight()

#change value 
def updateWeight():
	global wH
	global wY
	#update i to h
	for i in range(0,len(wH)):
		for j in range(0,len(wH[0])):
			wH[i][j]+=wIn[j][i]
	#update h to y
	for i in range(0, len(wY)):
		for j in range(0,len(wY[0])):
			wY[i][j]+=dH[i][j]

#cek error
def cekError():
	er=[]
	for i in range(0,len(outY)):
		tmp=0
		for j in range(0,len(outY[0])):
			tmp+=(yT[i][j]-outY[i][j])**2
		tmp*=0.5
		er.append(tmp)
	tmp=0
	for i in range(0,len(er)):
		tmp+=er[i]
	tmp*=0.5
	return tmp

def cekSSE():
	er=[]
	for i in range(0,len(outY)):
		tmp=0
		for j in range(0,len(outY[i])):
			tmp+=(yT[i][j]-outY[i][j])**2
		er.append(tmp)
	tmp=0
	for i in range(0,len(er)):
		tmp+=er[i]
	return tmp
def dataUji(inputan):
	global outY
	global outH
	outY=[]
	outH=[]
	for h in range(0,layer):
		oH=[]
		for i in range(0,len(inp)):
			tmp=0
			for j in range(0,len(inp[i])):
				tmp+=inputan[i][j]*wH[h][j]
				tmp+=b[0]
			tmp=fungsiAktifasi(tmp)
			oH.append(tmp)
		outH.append(oH)
	#output layer
	for i in range(0, len(outH[0])):#6
		oY=[]
		for j in range(0,Y):  #2
			tmp=0
			for k in range(0,len(outH)): #10
				z=outH[k][i]*wY[j][k]
				tmp+=z
				#print(f'tmp : {tmp}')
				#print(f"outH :{outH[k][i]} , {wY[j][k]}= {z}")
			tmp+=b[1]
			tmp=fungsiAktifasi(tmp)
			oY.append(tmp)
		outY.append(oY)

def main(val):
	#insertB()
	#insertW()
	global wH
	global wY
	global b
	b=[0,0]
	wH=[[-0.7,0.3],[0.2,0.3]]
	wY=[[0.3,-0.6]]
	maxIt=val
	forwardPass()
	er=cekError()
	maxI=len(inp)
	i=0
	it=0
	sse=cekSSE()
	print(sse)
	while(it<maxIt):
		if sse<0.1:
			break
		if i==maxI:
			i=0
		backwardPass(i)
		forwardPass()
		er=cekError()
		sse=cekSSE()
		i+=1
		it+=1
		print(f'it : {it} , sse: {sse} , error : {er}')
	print("bobot akhir : ",wY)

	#how to run
	#->import bp
	#->bp.main(maxIteration)
	#maxIteration is a variable number of maximum iteration etc. 100000
	#->bp.main(100000)