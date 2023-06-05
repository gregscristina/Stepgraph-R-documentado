#'
#' @title Cross-Validation for Stepwise Gaussian Graphical Model
#'
#' @description \code{cv.stepGraph} implements the cross-valiation procedure for the stepwise gaussian graphical algorithm.
#'
#' @param x Data matrix (of size n x p).
#' @param fold Number of folds for the cross-validation procedure.
#' @param alpha_f_min Minimum threshold value for the cross-validation procedure.
#' @param alpha_f_max Minimum threshold value for the cross-validation procedure.
#' @param n_alpha Number of elements in the grid for the cross-validation.
#' @param nei.max Maximum number of variables in every neighborhood.
#'
#' @return A list
#'
#' @export
#'
#' @author Anthony-Alexander Christidis, \email{anthony.christidis@stat.ubc.ca}
#'
#'
cv.stepGraph = function(x, fold, alpha_f_min, alpha_f_max, n_alpha, nei.max){

  cv.part = function(n, k)
  { # Para que haga un in and out
    ntest = floor(n/k) #floor()arredonda para inteiro # determinar tamanho do conjunto de teste
    ntrain = n - ntest #determina tamanho do conjunto de treinamento
    ind = sample(n) #A função sample é usada para criar um vetor aleatório ind contendo números de 1 a n.
    trainMat = matrix(NA, nrow=ntrain, ncol=k) #As matrizes trainMat e testMat são inicializadas com valores NA utilizando a funçao matrix(), 
    testMat = matrix(NA, nrow=ntest, ncol=k)    #com as dimensões adequadas para armazenar os conjuntos de treinamento e teste.
    nn = 1:n             #O vetor nn é criado com números de 1 a n.
    for (j in 1:k) {
      sel = ((j-1)*ntest+1):(j*ntest) # calcula os índices dos dados que serão selecionados para teste no fold atual. Ele define uma sequência de índices que corresponde aos dados do fold atual.
      testMat[,j] = ind[sel ] #atribui os dados selecionados para teste na matriz 
      sel2 = nn[ !(nn %in% sel) ] #calcula os índices dos dados que serão selecionados para treinamento no fold atual. Ele seleciona os índices que não estão presentes nos índices de teste.
      trainMat[,j] = ind[sel2] #atribui os dados selecionados para treinamento na matriz
    }
    return(list(trainMat=trainMat,testMat=testMat))
  }


  n = nrow(x)
  p = ncol(x)
  part.list = cv.part(n, fold)



  ## The grid

  alpha_seq = seq(alpha_f_min,alpha_f_max,length=n_alpha)#É criada uma sequência de valores entre alpha_f_min e alpha_f_max com tamanho n_alpha utilizando a função seq. Essa sequência será usada como os valores de alpha_f no grid de valores.
  alpha_f = rep(alpha_seq,2)
  alpha_b = c(0.5*alpha_seq,alpha_seq)
  alpha.grid <- cbind(f=alpha_f, b=alpha_b )
  alpha.grid  <- as.data.frame(alpha.grid )
  #A sequência alpha_seq é replicada duas vezes para formar o vetor alpha_f. 
  #O vetor alpha_b é formado multiplicando a sequência alpha_seq por 0.5 e concatenando-a com a sequência original alpha_seq. Esses vetores alpha_f e alpha_b são combinados em uma matriz alpha.grid usando a função cbind.

  #A matriz alpha.grid é convertida em um objeto do tipo data frame usando a função as.data.frame.

  


  loss.re = matrix(0, nrow=nrow(alpha.grid), ncol=fold)
  #Uma matriz loss.re é criada com dimensões nrow(alpha.grid) x fold, preenchida com zeros. Essa matriz será usada para armazenar os valores de perda (loss) calculados durante a validação cruzada.
  #
  #
  for (k in 1:fold)
  {
    x.train = x[part.list$trainMat[, k], ]
    varepsilonlist = lapply(1:nrow(alpha.grid),
                            function(i) stepGraph(x.train,alpha.grid$f[i],alpha.grid$b[i],nei.max))
    x.test = scale(x[part.list$testMat[, k], ])
    for (i in 1:nrow(alpha.grid)){
      if (length(varepsilonlist[[i]])==1){loss.re[i, k] = NA}
      else{
        beta = varepsilonlist[[i]][[2]]
        loss.re[i, k] = sum(colSums((x.test - x.test%*%beta)^2))}
    }
  }


  # Tabla de la funcion loss.mean para ver si hay un minimo

  loss.mean = apply(loss.re, 1, mean)

  alpha.grid$loss=loss.mean

  scatterplot3d::scatterplot3d(alpha.grid$f, y=alpha.grid$b,z=alpha.grid$loss)

  ind = which.min(loss.mean)
  alpha_f_opt = alpha.grid$f[ind]
  alpha_b_opt = alpha.grid$b[ind]
  CV.loss = alpha.grid
  colnames(CV.loss) = c("alpha_f", "alpha_b", "CV.loss")

  res = list(alpha_f_opt,alpha_b_opt,CV.loss)
  return(res)
}
