#!/usr/bin/env python
# coding: utf-8

#                                         Installing Numpy Library

# In[1]:


pip install numpy


#                                         Numpy Basic Array Manipulations

# In[2]:


import numpy as np


#                                    Multidimenional array objects (nd array)

# In[3]:


# An array containing zeros

np.zeros((2,2))


# In[4]:


# An array containing ones

np.ones((2,2))


# In[5]:


# A random array ( not always zeros, can be any number)

np.empty((2,2))


# In[6]:


# Converting a python list into a numpy array

a=["Ali",5,34,2]
b=np.array(a)
print(b)

""" Now checking type"""

type(b)


# In[7]:


# Creating an array in a given range (similar to python k loop ki range)

"""1st argument is the starting point, 2nd arg is ending point & 3rd arg is the num of steps """

np.arange(1,100,5)


# In[8]:


# Gives the shape of an array (size of array along each dimemsion)

x=np.ones((10,10))
x.shape


# In[9]:


# Changes the dimesnsion into n-dimension where n is the number of dimension you want

"""The size or dimension should remain same , For Eg: In x, the shape was 10x10 which is equal to 100.
    If you reshape x, the product after reshape should be 100 or it won't work"""

x=np.ones((10,10))
x.reshape((2,2,5,5))


# In[10]:


# Checking data type

y=np.empty((2,2))
y
y.dtype


# In[11]:


# Gives the dimension

x=np.empty((2,5,4))
print(x)

"""The dimension of x is"""

x.ndim


#                                       
#                                       Arithmetic Operation with nd arrays 

# In[12]:


# Creating a random array
a=np.random.randn(10)
a


# In[13]:


b=np.random.randn(10)
b


# In[14]:


"""All the operations are index to index"""


# Adding two arrays

a+b


# In[15]:


# Sbbtracting them

a-b


# In[16]:


# Multiplying them

a*b


# In[17]:


# dividing them
a/b


# In[18]:


# Comparison of arrays

a==b


# In[19]:


# Checking if thethe array is greater than zero

"""It will return the answer of each index in array in boolean form"""

a>0


# In[20]:


# fetching the values greater than zero

"""We pass the whole statement as an index in the same array"""

a[a>0]


#                                                Indexing

# In[21]:


# positive index

x=np.array([1,2,3,4,5])
x

""" The indexing is same as that of simple python list. The only difference 
    is that the indices here are of a numpy array and not of a list """


x[1]


# In[22]:


# negative index

x[-2]


# In[23]:


# Boolean indexing

""" This is to extract out elements by giving conditions in index """

x[x>2]


# In[24]:


# Fancy indexing

"""We passed a list as an index. Each index in this list gives the output that 
    corresponds to the value of that index in the original array"""

x[[2,3,4]]


#                                                     Slicing

# In[25]:


# Making a random array of order 5x5 for slicing

a=np.random.randn(5,5)
a


# In[26]:


# Slicing of rows

"""The slicing is same as of normal python string or list.
    It is slicing of rows if we consider this array as a matrix."""

""" This gives first row"""
    
a[0]


# In[27]:


# slicing of coloumns

""" This gives values from coloumn 2 to coloumn 5"""

a[::,1:4] 


# In[28]:


# SLicing of single coloumn w.r.t it's row

'''This gives value of 5th coloumn in 1st row'''

a[0][4]   


# In[29]:


# Creating a square with ones as borders and zeros as center

s=np.ones((5,5))

print("Original matrix:")
print()
print(s)

"""The first slicing represents rows whereas slicing after comma represents coloumns"""

s[1:4,1:4]=0
s


# In[30]:


# Creating a chess board

"""This is done in 2 steps. In step 1 the odd rows and coloumns were made 0 using steps.
    In step 2, the even rows and coloumns were made 0 using steps"""

w=np.ones((8,8))
w

""" Step 1 """
w[0:9:2,0:9:2]=0
chessBoard=w

""" Step 2 """

chessBoard[1:9:2,1:9:2]=0
print(chessBoard)


#                                         Element-wise Array Functions

# In[31]:


x=np.array([1,2,4,6,44])
x


# In[32]:


# Gives exponent

""" All the functions gives element-wise results """

np.power(x,3) # Gives cube of each element in array


# In[33]:


# Getting square root

np.sqrt(x)


# In[34]:


# Gives square

np.square(x)


# In[35]:


# Gives sine angle of each element

np.sin(x)


# In[36]:


# Gives cosine angle of each element

np.cos(x)


# In[37]:


# Gives tangent angle of each element

np.tan(x)


#                                         Where Function

# In[38]:


""" The format of where function goes like this,
    where(condition,action(if True),action otherwise) """

# It kind of works like an if condition where the condition is 1st argument with automatic if
# and 2nd argument is the fulfilment of the condition if true and 3rd argument is the else part.

salary=np.array([0,-1,50000,44000])
np.where(salary<=0,"Not Ok","Ok")


#                                     Mathematical & Statistical Functions

# In[39]:


# Getting mean

i=np.array([1,82,3,6])
np.mean(i)


# In[40]:


# Gives commulative sum

"""" Each element afterwards is equal to the sum of its previous elements """

np.cumsum(i)


# In[41]:


# Gives commulative product

"""" Each element afterwards is equal to the product of its previous elements """

np.cumprod(i)


# In[42]:


# Checking condition for each index

x=i>6
x


# In[43]:


# Gives sum of all elements

np.sum(i)


# In[44]:


# Checks if any element is True

np.any(x)


# In[45]:


# Checks if all element are True

np.all(x)


# In[46]:


# Sorts the array

np.sort(i)


# In[47]:


# Gives unique elements

np.unique(i)


#                                                 File Input Output

# In[48]:


# Saving an array in a file

arr1=[1,2,3,4]
arr2=[2,3,6,8,9]
arr3=[3,4,66,2,1]
np.save("singleSave.npy",arr2)


# In[49]:


# loading a saved file

"""This gives the result in a dictionary. All the arrays are stored as keys and 
    can be fetched in the same manner as a key of dictionary is fetched """

np.load("singleSave.npy")

#    Since, it's a single key (array) here, therefore, the result is given directly. It 
#    would have been fetched by a dictionary key fetching method otherwise !


# In[50]:


# saving multiple arrays in a single file 

np.savez("MultiSave",arr1,arr2,arr3)
myDict=np.load("MultiSave.npz")
print(myDict)


#                                             Linear Algebra Functions

# In[51]:


x=np.random.randn(3,3)
y=np.random.randn(3,3)
print("x=","\n",x)
print("y=","\n",y)


# In[52]:


# Dot product of 2 matrices

np.dot(x,y)


# In[53]:


# Transpose of a matrix

np.transpose(x)


# In[54]:


# Trace of a matrix

np.trace(x)


# In[55]:


# diagonal of a matrix

np.diag(x)


# In[56]:


# Eigen values and correseponding vectors of  matrix

from numpy import linalg as LA
values,vectors=LA.eig(x)
print("Eigen values are:","\n",values)
print()
print("Eigen vectors are:","\n",vectors)


# In[57]:


# Gives inverse of a matrix

LA.inv(x)


# In[58]:


# Gives rank of  a matrix

LA.matrix_rank(x)


# In[59]:


# Gives determinant of a matrix

LA.det(x)


# In[60]:


# Gives adjoint of a matrix

np.matrix.getH(x)


# In[61]:


# Gives conjugate of a matrix

np.conjugate(x)


#                                                 Pseudo Random Numbers

# In[62]:


# To initialize

np.random.seed(7)


# In[63]:


# Normal distributuion

np.random.normal(size=(3,3))


# In[64]:


# Uniform distributuion

np.random.uniform(size=(3,3))


# In[65]:


# Chi Square distributuion

np.random.chisquare((5,5))


# In[66]:


# Gamma distributuion

np.random.gamma((1,1))


#                                      Numpy Advanced Array Manipulations

#                                             Reshape Function

# In[72]:


# Reshape basically changes the order of elements in matrix by manipulating axes and dimensions.
# The count of elements and elements themselves remains same. It is simply change of way a matrix 
# is written by manipulation of dimensions.

a=np.random.randn(4,5)
print("a:","\n",a)

"""The product of dimenssions should be equal when reshaping"""

a.reshape(2,2,5)


# In[79]:


# Coloumn major order or Fortran Order

a.reshape((2,2,5),order="F")


# In[78]:


# Row major order or C Order


a.reshape((2,2,5),order="C")


# In[82]:


# Converting in 1 dimension

x=a.ravel()
print(x)
print()
y=a.flatten()
print(y)


# In[86]:


""" -1 leaves it upto numpy to complete the rest of the dimension """

a.reshape(10,-1)


#                                              Concatenate Function

# In[91]:


# Row-wise concatenation

x=np.array(([1,2,3],[2,4,6]))
y=np.array(([11,12,13],[12,14,16]))

""" It requires an axis. It can either be 0 or 1. 0 represents row-wise concatenation.The number 
    of  rows increases whereas the number of coloumns remains same in row-wise concatenation """

np.concatenate((x,y),axis=0)


# In[92]:


# Coloumn-wise concatenation

x=np.array(([1,2,3],[2,4,6]))
y=np.array(([11,12,13],[12,14,16]))

""" It requires an axis. It can either be 0 or 1. 1 represents coloumn-wise concatenation.The number 
    of  coloumns increases whereas the number of rows remains same in coloumn-wise concatenation """

np.concatenate((x,y),axis=1)


# In[98]:


# Default Row-wise concatenation

np.vstack((x,y))


# In[97]:


# Default Coloumn-wise concatenation

np.hstack((x,y))


#                                                Split Function

# In[121]:


# Splitting of an array into n arrays

x=np.array(([1,2,3,2,4,6,6,9,0,5,3]))

""" The first argument is of the array to be split and second is for the positions on which split occurs"""

np.split(x,[2,5,7])


# In[120]:


# Coloumn-wise split

x=np.array(([1,2,3,5,7,9,3],
            [2,4,6,8,10,12,14]))

""" It is a coloumn-wise split. First split of both arrays in the tuple occurs at 
    coloumn 2 and second split occurs at coloumn 5 for both arrays in the tuple. 
    The remaining elements of a single array is splitted in an array for each 
    individual array in the tuple """

np.split(x,[2,5],axis=1)


# In[124]:


# Row-wise split

x=np.array(([1,2,3,5,7,9,3],
            [2,4,6,8,10,12,14]))

""" It is a row-wise split. First split of both arrays in the tuple occurs at row 2
    and second split occurs at row 5 for both arrays in the tuple. The remaining 
    elements of every array is splitted in individual arrays for each initial array"""

np.split(x,[2,5],axis=0)


# In[127]:


# Default row-wise split

np.vsplit(x,[2,5])


# In[134]:


# Default coloumn-wise split

np.hsplit(x,[2,5])


#                                                 Repeat Function

# In[140]:


# Repetition of each element in same array for n number of times

x=np.array(([1,2,3,5,7,9,3]))

""" 1st argument represents the array to be repeated whereas 2nd argument takes
    the number of times each element should be repeated as a parameter  """

np.repeat(x,4)


#                                             Tile Function

# In[141]:


# Creating a copy of an array in same array as individual elements

x=np.array(([1,2,3,5,7,9,3]))

""" 1st argument represents the array to be copied whereas 2nd argument takes
    the number of times the array should be copied as a parameter  """

np.tile(x,3)


# In[ ]:





# In[ ]:




