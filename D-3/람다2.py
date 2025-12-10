#람다 => 모든 언어 채택, 스파크
#킆라우드 => 여러대의 서버를 서로 연결시켜서 하나의 서버처럼 

#람다 - 파이썬은 한줄짜리 함수

#x값 받아서 5보다 크면 True 를 작으면 False 를 뱐환하는 함수이다.
def greater(x): #콜백함수
    return x>5

#filter(함수, 이터러블데이터) 이터럽블데이터의 요소 하나씩을 함수에 전달해서 저함수가 True를 반환하면
# 값을 전달해주고 False면 안준아.

a = [1,2,3,4,5,6,7,8,9,10]
#filter(greater, a) - iterable 객체

print( list(filter(greater, a)))

for i in filter(greater, a):
    print(i)
#람다 만든 이유 
#프로그램에서 필요한 요소들이 정해져 있다. 검색하고 정렬 ......
#파이썬의 람다는 한줄만 가능 => 자바스크립트같은 경우에는 람다가 여러줄 들어가도 됨 
for i in filter(lambda x: x>5, a):
    print(i)



