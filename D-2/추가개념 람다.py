calc = lambda x, y: x+y 
print( calc( 8, 7)) 

a = [1,2,3,4,5,6,7,8,9,10]
#특정 조건에 해당하는 짝수인것만 혹은 5보다 큰것만 

#filter
#filter(함수, 반복타입) 

#작수만 선택하고자 할때 
#callback함수가 호출자가 시스템임 - 잠깐 쓰고 버린다. 

#1.
for i in filter( lambda x:x%2==0 , a):
    print(i)

#2. 
def isEven(x):
    return x%2==0

for i in filter( isEven, a):
    print(i)
