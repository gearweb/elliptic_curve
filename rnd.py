import random
num = set()
f= open("numbers.txt","w+")
while len(num) < 3900:
    rand = random.randint(1,10000)
    num.add(rand)
    print(f"{rand:04d}")

for i in num:
    f.write(f"\"{i:04d}\"\r\n")
f.close() 