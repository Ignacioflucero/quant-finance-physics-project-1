import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data2 = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01")
prices = data2["Close"].squeeze() #np.squeeze descartar dimensiones redundantes 
returns = prices.pct_change().dropna().squeeze()
mu = returns.mean()
sigma = returns.std()

#Parametros de simulacion
S0 = prices.iloc[-1] #precio inicial
print("Precio Inicial=",S0)
T = 252
dt = 1
N_sim = 100
trayectorias = []
precios_finales = []
#Monte Carlo

plt.figure(figsize=(10,6))
for i in range(N_sim):
    S = [S0]
    for t in range(T):
        Z = np.random.normal()
        next_price = S[-1] * np.exp((mu - 0.5*sigma**2)+ sigma*Z) #Euler-Maruyama integración numérica de un proceso estocástico
        S.append(next_price)
    precios_finales.append(S[-1])
    trayectorias.append(S)
    plt.plot(S)
plt.title("Monte Carlo Simulacion")
plt.xlabel("Tiempo")
plt.ylabel("Precio")
plt.grid()
plt.savefig("montecarlo.png")
plt.show()

trayectorias = np.array(trayectorias)
media_trayec = np.mean(trayectorias, axis=0) #el axis hace un promedio entre simulaciones para cada tiempo
desv_trayec = np.std(trayectorias, axis=0)

plt.figure(figsize=(10,6))
plt.plot(media_trayec)
plt.title("Trayectoria promedio")
plt.xlabel("Tiempo")
plt.ylabel("Precio promedio")
plt.grid()
plt.savefig("preciopromedio.png")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(desv_trayec)
plt.title("Crecimiento de incertidumbre temporal")
plt.xlabel("Tiempo")
plt.ylabel("Desviación estándar")
plt.savefig("crecimientotemporal.png")
plt.grid()

plt.show()

precios_finales = np.array(precios_finales)
plt.figure(figsize=(10,6))
plt.hist(precios_finales, bins=30)
plt.title("Distribución de precios finales")
plt.xlabel("Precio final")
plt.ylabel("Frecuencia")
plt.grid()
plt.savefig("distribuciondeprecios.png")
plt.show()

p5 = np.percentile(precios_finales, 5)
print("Percentil 5% =", p5)
VaR = S0 - p5 #Riesgo Financiero ; valor en riesgo
print("VaR 95% =", VaR) #Existe un 5% de probabilidad de perder más de ..... USD

X = []
y = []
for i in range(3, len(returns)):  #sliding window
    X.append([float(returns.iloc[i-3]),float(returns.iloc[i-2]),float(returns.iloc[i-1])])
    y.append(float(returns.iloc[i]))
X = np.array(X)
y = np.array(y)
#print(X.shape)
#print(y.shape)
split = int(0.8 * len(X)) # 20% prueba , 80% entrenamiento
X_train = X[:split] #Entrenamiento
X_test = X[split:] #Prueba
y_train = y[:split]
y_test = y[split:]

modelo = LinearRegression() #y^​=β0​+β1​x1​+β2​x2​+β3​x3​
modelo.fit(X_train, y_train) #Encuentra los betas
predicciones = modelo.predict(X_test)

mse = mean_squared_error(y_test,predicciones) #MSE:Error cuadrático medio
print("MSE =", mse) #saber lo que pasó ayer ayuda poco a predecir mañana.
#Visualmente
plt.figure(figsize=(10,5))
plt.plot(y_test[:100], label="Real")
plt.plot(predicciones[:100], label="Predicción")
plt.title("Real vs Prediccion")
plt.legend()
plt.grid()
plt.savefig("comparativarealvspredic.png")
plt.show()