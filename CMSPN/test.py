# define the function blocks
def zero():
    print ("You typed zero.")

def sqr():
    print ("n is a perfect square")

def even():
    print ("n is an even number")

def prime():
    print ("n is a prime number")

# map the inputs to the function blocks
options = {0 : zero,
           1 : sqr,
           4 : sqr,
           9 : sqr,
           2 : even,
           3 : prime,
           5 : prime,
           7 : prime,
}
if 1==1:
  a=2
else:
  a=3
print (a+1)