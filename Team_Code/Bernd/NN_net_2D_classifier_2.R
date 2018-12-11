# predictor variables
X <- matrix(c(
  0.1, 0.9,
  0.3, 0.7,
  0.2, 0.9,
  0.3, 0.2,
  0.1, 0.2,
  0.2, 0.1,
  0.7, 0.2,
  0.7, 0.5,
  0.9, 0.9
),
ncol = 2, #number of input variables
byrow = TRUE
)

# hidden neurons
hn <- 12

# observed outcomes
y <- c(0, 0, 1, 0, 0, 0, 1, 1, 1)

# print the data so we can take a quick look at it
t_data <- as.data.frame(cbind(X, as.character(y)))
names(t_data) <- c("cX","cY","classif")
t_data

ggplot(data = t_data, aes(x = cX, y = cY, colour =  classif , shape = classif )) +
  geom_point(size = 4 )  + 
  scale_color_manual(values=c("chartreuse", "blue4" ))

# generate a random value between 0 and 1 for each
# element in X.  This will be used as our initial weights
# for layer 1
rand_vector <- runif(ncol(X) * hn)

# convert above vector into a matrix
rand_matrix <- matrix(
  rand_vector,
  nrow = ncol(X), #number of input columns / neurons
  ncol = hn, #number of hidden neurons. Error in original script where they just say that the number of hidden neurons = number of input points.
  byrow = TRUE
)

# create random weights for W2: hidden layer -> output
rand_matrix_2 <- matrix(runif(hn), ncol = 1)

# this list stores the state of our neural net as it is trained
my_nn <- list(
  # predictor variables
  input = X,
  # weights for layer 1
  weights1 = rand_matrix,
  # weights for layer 2
  weights2 = rand_matrix_2,
  # actual observed
  y = y,
  # stores the predicted outcome
  output = matrix(
    rep(0, times = hn),
    ncol = 1
  )
)
 
  

#' the activation function
sigmoid <- function(x) {
  1.0 / (1.0 + exp(-x))
}

#' the derivative of the activation function. x is in fact sigmoid(x). The derivative is written in function of the activation function.
sigmoid_derivative <- function(x) {
  x * (1.0 - x)
}


# The goal of the neural network is to find weights for each layer that minimize the result of the loss function.

#' feedforward
feedforward <- function(nn) {
  nn$layer1 <- sigmoid(nn$input %*% nn$weights1)  # sigmoid(N %*% rand_matrix) = OUTPUT layer 1 voor elk punt: 1 rij met uitkomsten voor de weights: 4 hidden neurons
  nn$output <- sigmoid(nn$layer1 %*% nn$weights2) # sigmoid(sigmoid(N %*% rand_matrix) %*% rand_matrix_2) 
  nn
}

#' loss function
loss_function <- function(nn) {
  sum((nn$y - nn$output) ^ 2)
}

# sum((y - sigmoid(sigmoid(X %*% rand_matrix) %*% rand_matrix_2) )  ^ 2)

#' backprop
 
backprop <- function(nn) {
  
  # application of the chain rule to find derivative of the loss function with 
  # respect to weights2 and weights1
  # d_weights is the adjustment of the weight vectors
  
  d_weights2 <- ( t(nn$layer1) %*% # t() = transpose. nn$layer1 is de uitkomst na layer 1, voor elk punt. Normaal: 1 punt = 1 rij. Elke rij heeft sowieso evenveel elementen als hidden neurons
                (2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) )
              # `2 * (nn$y - nn$output)` is the derivative of the loss function
  
  
  d_weights1 <- ( 2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) %*%  t(nn$weights2)
  
  d_weights1 <- d_weights1 * sigmoid_derivative(nn$layer1)
  d_weights1 <- t(nn$input) %*% d_weights1
  
  # update the weights using the derivative (slope) of the loss function
  nn$weights1 <- nn$weights1 + d_weights1 
  nn$weights2 <- nn$weights2 + d_weights2
  
  nn
}

# number of times to perform feedforward and backpropagation
epochs <- 10000

# data frame to store the results of the loss function.
# this data frame is used to produce the plot in the 
# next code chunk
loss_df <- data.frame(
  iteration = 1:epochs,
  loss = vector("numeric", length = epochs)
)

for (i in seq_len(epochs)) {
  my_nn <- feedforward(my_nn)
  my_nn <- backprop(my_nn)
  
  # store the result of the loss function.  We will plot this later
  loss_df$loss[i] <- loss_function(my_nn)
}

# print the predicted outcome next to the actual outcome

Score <- data.frame(
  "Predicted" = round(my_nn$output, 3), # actual prediction is not 0/1 but on a continuous scale
  "Classify" = round(my_nn$output, 0),  # if we have to make a decision, we round the decimal to a 0/1 integer
  "Actual" = y
)

# calculate accuracy of the network as percentual error on training set

Accuracy <- ( 1 - sum(abs(Score$Classify - Score$Actual)) / nrow(Score) ) * 100

print(paste0("The accuracy of the network on the trainingsset is ", round(Accuracy,1), "%"))



library(ggplot2)

ggplot(data = loss_df, aes(x = iteration, y = loss)) +
  geom_line()


# Validate unseen point

V <- matrix(c(0.2, 0.7))

# slightly modify nn function to take in point V

feedforward_validate <- function(nn,Val) {
  nn$layer1 <- sigmoid(t(Val) %*% nn$weights1)  # sigmoid(N %*% rand_matrix) = OUTPUT layer 1 voor elk punt: 1 rij met uitkomsten voor de weights: 4 hidden neurons
  nn$output <- sigmoid(nn$layer1 %*% nn$weights2) # sigmoid(sigmoid(N %*% rand_matrix) %*% rand_matrix_2) 
  nn$output
}

# gives the outcome of point V

feedforward_validate(my_nn,V)


# Create lattice of unseen points


lX <- seq(0.1, 1.0, length = 100)
lY <- seq(0.1, 1.0, length = 100) 

NN_GRID <- setNames(data.frame(matrix(ncol = 3, nrow = 0)), c("lX", "lY", "nn_output"))

for (aX in lX) {
  for (aY in lY) {
    aV <- matrix(c(aX, aY))
    nn_output <- round(feedforward_validate(my_nn,aV),3)
    NN_GRID <- rbind(NN_GRID, c(aX,aY,nn_output))
    
  }
  
}

names(NN_GRID) <- c("lX","lY","nn_output")

# draw the contour map

ggplot(data = NN_GRID, aes(x = lX, y = lY, fill = nn_output)) +
  geom_tile() + 
  scale_fill_distiller(palette="Spectral", na.value="white") + 
  theme_bw()





