#'
#' @importFrom stats cor cov lm residuals var
#' @importFrom utils combn
#'
#' @title Stepwise Gaussian Graphical Model
#'
#' @description \code{stepGraph} computes the output for the stepwise gaussian graphical algorithm.
#'
#' @param x Data matrix (of size n x p). ---------------------------------A)
#' @param alpha_f Forward threshold.  ------------------------------------B)
#' @param alpha_b Backward threshold. ------------------------------------C)
#' @param nei.max Maximum number of variables in every neighborhood. -----D)
#' 
#' 
#' 
#' =================================================================================================================================================
#' =====================================================================MEUS COMENTARIOS ===========================================================
#' A) matriz de dados
#' DUVIDA ---------B)limiar para o passo de adição - Ele representa um limite para a estatística de teste que determina se uma variável deve ser adicionada ao conjunto de variáveis selecionadas.
#' DUVIDA----------C)limiar para o passo de remoção - Ele representa um limite para a estatística de teste que determina se uma variável deve ser removida do conjunto de variáveis selecionadas.
#' D)(número máximo de variáveis em cada vizinhança)- Esse comentário indica que o parâmetro nei.max define o número máximo de variáveis permitidas em cada vizinhança.
#'    Uma vizinhança é um conjunto de variáveis que são consideradas juntas em um determinado estágio do algoritmo. 
#'    Esse parâmetro controla a complexidade do modelo resultante, limitando o número máximo de variáveis nas vizinhanças.
#' =================================================================================================================================================
#' ==================================================================================================================================================
#' 
#' 
#'
#' @return A list with the values: \cr \cr-------------------------------------------------------E)
#' \code{Edges_A}: Estimated set of edges, \cr \cr-----------------------------------------------F)
#' \code{Omega}: Estimated precision matrix, \cr \cr---------------------------------------------G)
#' \code{Adj_mat}: A (p x p) zero diagonal matrix. The non-zero off-diagonal elements------------H)
#' corresponds with the order in which the ordered pair (i,j) is selected in the
#' forward step, and \cr \cr
#' \code{k_stop}:  Final step in the iteration.--------------------------------------------------I)
#' @export
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#' 
#' ===================================================================================================================================================
#'=======================================MEUS COMENTARIOS ===========================================================================================
#' E)Essa linha indica que a função retorna uma lista com os seguintes valores.
#' F)Essa linha descreve que Edges_A é o conjunto estimado de arestas (conexões) entre as variáveis.
#' G) descreve que Omega é a matriz de precisão estimada.
#'    A matriz de precisão é a matriz inversa da matriz de covariância e é usada para descrever as relações entre as variáveis em um modelo gráfico.
#' H)Essa linha descreve que Adj_mat é uma matriz de dimensões (p x p) com elementos diagonais zero. 
#'    Os elementos fora da diagonal não nula correspondem à ordem em que o par ordenado (i, j) é selecionado no passo de adição.
#' I) Essa linha descreve que k_stop representa o último passo na iteração do algoritmo.
#'===================================================================================================================================================
#'====================================================================================================================================================
#'
#' 





stepGraph = function(x, alpha_f, alpha_b, nei.max){

  #Initialization

  if (length(nei.max)==0){nei.max=n-1}

  x = scale(x)

  n = dim(x)[1]

  p = dim(x)[2]

  Edges_I = combn(1:p,2) # Inactive set of ordered pair (i,j)

  Edges_A = matrix(0,2,dim(Edges_I)[2]) # Active set of ordered pair (i,j)

  Adj_mat = matrix(0,p,p)

  e = x # (n x p) matrix of regression residuals

  k = 1

  K = length(which(Edges_I[2,]>0))

  ######
  ######

  while ((k<=K)){

    # Forward Step

    # Compute Prediction Errors for
    # (i,j) in the inactive set Edges_I

    f_ij = rep(0,length(which(Edges_I[2,]>0)))

    for (h in which(Edges_I[2,]>0)){
      i = Edges_I[1,h]
      j = Edges_I[2,h]

      f_ij[which(h==which(Edges_I[2,]>0))] = cor(e[,i],e[,j])
    }

    # Select (i,j) that max(|f_ij|)

    if (length(abs(f_ij)[(abs(f_ij)>=alpha_f)])==0) {break}

    else{

      Edge_opt_forward = which(Edges_I[2,]>0)[which.max(abs(f_ij))]
      f_ij_opt_forward = max(abs(f_ij))

      Edges_A[,Edge_opt_forward] = Edges_I[,Edge_opt_forward]
      Edges_I[,Edge_opt_forward] = c(0,0)

      i_f = Edges_A[1,Edge_opt_forward]
      j_f = Edges_A[2,Edge_opt_forward]

      Adj_mat[i_f,j_f] = Adj_mat[j_f,i_f] = Adj_mat[i_f,j_f] + k

      # Update Prediction Errors for (i_f,j_f)

      n_i_f = c(Edges_A[1,which(Edges_A[2,]==i_f)],
                Edges_A[2,which(Edges_A[1,]==i_f)])

      if (length(n_i_f)>= nei.max){break}
      else{e[,i_f] = residuals(lm(x[,i_f] ~ x[,n_i_f]))}

      n_j_f = c(Edges_A[1,which(Edges_A[2,]==j_f)],
                Edges_A[2,which(Edges_A[1,]==j_f)])

      if (length(n_j_f)>= nei.max){break}
      else{e[,j_f] = residuals(lm(x[,j_f] ~ x[,n_j_f]))
      }


      ######

      # Backward-Step

      # Compute Prediction Errors r_i and r_j for
      # (i,j) in the active set Edges_A

      b_ij = rep(0,length(which(Edges_A[2,]>0)))

      for (l in which(Edges_A[2,]>0)){
        i = Edges_A[1,l]
        j = Edges_A[2,l]

        n_i = c(Edges_A[1,which(Edges_A[2,]==i)],
                Edges_A[2,which(Edges_A[1,]==i)])
        n_i = n_i[-which(n_i==j)]

        if(length(n_i)>0){r_i = residuals(lm(x[,i] ~ x[,n_i]))}
        else{r_i = x[,i]}

        n_j = c(Edges_A[1,which(Edges_A[2,]==j)],
                Edges_A[2,which(Edges_A[1,]==j)])
        n_j = n_j[-which(n_j==i)]

        if(length(n_j)>0){r_j = residuals(lm(x[,j] ~ x[,n_j]))}
        else{r_j = x[,j]}

        b_ij[which(l==which(Edges_A[2,]>0))] = cor(r_i,r_j)
      }

      # Select (i,j) that min(|b_ij|)<= alpha_b

      Edge_opt_backward = which(Edges_A[2,]>0)[which.min(abs(b_ij))]
      f_ij_opt_backward = min(abs(b_ij))

      # Update active set of edges Edges_A

      if (f_ij_opt_backward<=alpha_b){

        i_b = Edges_A[1,Edge_opt_backward]
        j_b = Edges_A[2,Edge_opt_backward]

        Edges_I[,Edge_opt_backward] = Edges_A[,Edge_opt_backward]
        Edges_A[,Edge_opt_backward] = c(0,0)

        Adj_mat[i_b,j_b] = Adj_mat[j_b,i_b] = 0

        # Update Prediction Errors for (i_b,j_b)

        n_i_b = c(Edges_A[1,which(Edges_A[2,]==i_b)],
                  Edges_A[2,which(Edges_A[1,]==i_b)])

        if(length(n_i_b)>0){e[,i_b] = residuals(lm(x[,i_b] ~ x[,n_i_b]))}
        else{e[,i_b] = x[,i_b]}

        n_j_b = c(Edges_A[1,which(Edges_A[2,]==j_b)],
                  Edges_A[2,which(Edges_A[1,]==j_b)])

        if(length(n_j_b)>0){e[,j_b] = residuals(lm(x[,j_b] ~ x[,n_j_b]))}
        else{e[,j_b] = x[,j_b]}

      }

      ######

      k = k+1
      if (f_ij_opt_forward<alpha_f) {break}
    }}

  k_stop = k

  # Compute Prediction Errors and betas

  vareps = x
  beta = matrix(0,p,p)

  for (e in which(Edges_A[2,]>0)){
    i = Edges_A[1,e]
    j = Edges_A[2,e]

    n_i = c(Edges_A[1,which(Edges_A[2,]==i)],
            Edges_A[2,which(Edges_A[1,]==i)])

    if(length(n_i)>0){vareps[,i] = residuals(lm(x[,i] ~ x[,n_i]))}
    else{vareps[,i] = x[,i]}

    n_j = c(Edges_A[1,which(Edges_A[2,]==j)],
            Edges_A[2,which(Edges_A[1,]==j)])

    if(length(n_j)>0){vareps[,j] = residuals(lm(x[,j] ~ x[,n_j]))}
    else{vareps[,j] = x[,j]}

    beta[i,j] = -cov(vareps[,i],vareps[,j])/(var(vareps[,j]))

    beta[j,i] = -cov(vareps[,i],vareps[,j])/(var(vareps[,i]))

  }

  # Compute the precision matrix

  Omega = matrix(0,p,p)

  diag(Omega) = apply(vareps,2,var)^(-1)

  for (e in which(Edges_A[2,]>0)){
    i = Edges_A[1,e]
    j = Edges_A[2,e]

    Omega[i,j] = cov(vareps[,i],vareps[,j])*Omega[i,i]*Omega[j,j]
    Omega[j,i] = Omega[i,j]
  }

  return(list(vareps=vareps, beta=beta, Edges_A=Edges_A,
              Adj_mat=Adj_mat, k_stop=k_stop, Omega=Omega))
}
