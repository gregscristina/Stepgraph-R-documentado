
rm(list=ls()) # Clear memory


# Verificar se o pacote MASS está instalado
if (!require(MASS)) {
  install.packages("MASS")
  library(MASS)
}

library(MASS) # mvrnorm
#source(file='StepGraph.R')
source(file='C:\\Users\\Cliente\\OneDrive\\Área de Trabalho\\estatistica\\QUARTO PERIODO\\SUPER2023- COLONA\\CODIGO R-STEP GRAPH\\Stepgraph-R-documentado')

# Setting the parameters
set.seed(1234567) # semente aleatória
phi = 0.4         # parâmetro autoregressivo
p = 20             # Dimension
n = 100           # Sample size

SigmaAR = function(p, phi){ # compute the covariance matrix
  Sigma = diag(p)
  for (i in 1:p) {
    for (j in 1:p) {
      Sigma[i,j] = phi^(abs(i-j))
    }
  }
  return(Sigma)
}
# gera a matriz de precisão (Omega) esparsa
Sigma = SigmaAR(p, phi)
Omega = solve(Sigma)  
Omega[abs(Omega) < 1e-5] = 0  

# Generate Data from a Gaussian distribution
X = list()
X = mvrnorm(n, mu=rep(0,p), Sigma)
X = scale(X)

print(X)
write.csv(X, file = "matriz_X.csv", row.names = FALSE)
# tem que salvar X em .csv


alpha_f = 0.24
alpha_b = 0.12
nei.max = 10

G = StepGraph(X, alpha_f = alpha_f, alpha_b = alpha_b, nei.max=nei.max)
G$Omega