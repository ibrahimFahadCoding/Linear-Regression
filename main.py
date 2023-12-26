import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([4, 5, 6, 3, 5, 7, 3, 9, 13, 14])


def gradient_descent(m_now, b_now, x, y, L):
  m_gradient, b_gradient = 0, 0
  n = len(x)
  
  for i in range(n):
    m_gradient += -(2/n) * x[i] * (y[i] - (m_now * x[i] + b_now))
    b_gradient += -(2/n) * (y[i] - (m_now * x[i] + b_now))
    
  m = m_now - m_gradient * L
  b = b_now - b_gradient * L
  return m,b


m = 0
b = 0
L = 0.0001
epochs = 1000
for i in range(epochs):
  if i % 50 == 0:
    print(f"Epoch: {i}")
  m, b = gradient_descent(m, b, x, y, L)
print(f"\nSlope: {m} \nY-Intercept: {b}")


plt.scatter(x,y,color="black")
plt.plot(x, m*x + b, color="red")
plt.savefig("MyLine.png")
plt.clf()

number = int(input("Enter Number: "))
print(m*number + b)




model = LinearRegression()
model.fit(x,y)
correlation = model.score(x,y)
m = model.coef_[0]
b = model.intercept_
print(f"Correlation: {correlation} \nSlope: {m} \nY Intercept: {b}")

plt.scatter(x,y,color="black")
plt.plot(x, m*x + b, color="blue")
plt.savefig("TheirLine.png")
plt.clf()

number = np.array([number]).reshape(-1,1)
print(model.predict(number)[0])














  

  
    
    
  
  
  
  
  
