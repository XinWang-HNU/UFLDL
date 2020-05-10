function [cost, Grad] = sparseAutoencoderCost (theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, Data)

% VisibleSize: the number of  input  Units (probably 64 ) 
 % hiddenSize: the number of Hidden Units (probably 25 ) 
 % lambda: Weight Decay Parameter
 % sparsityParam: The Desired average Activation for the Hidden Units (denoted in the Lecture
 % Notes by the Greek alphabet rho, which Looks like a Lower- Case  " P " ).
 % beta: Weight of sparsity Penalty TERM
 % Data: Our 64x10000 Data Matrix containing the training. So, Data (:, i) is the i- th training Example .
  
% The input theta is a Vector (Because minFunc the expects the Parameters to be a Vector). 
 % theta We first convert to the (W1, W2, b1, b2) Matrix / Vector format, SO that this 
 % Follows the Notation Convention of the lecture notes.

% the length vector into each layer weight matrix and bias vector-valued
W1 = reshape (theta ( 1 : hiddenSize * visibleSize), hiddenSize, visibleSize);
W2 = reshape (theta (hiddenSize * visibleSize + 1 : 2 * hiddenSize * visibleSize), visibleSize, hiddenSize);
b1 = theta ( 2 * hiddenSize * visibleSize + 1 : 2 * hiddenSize * visibleSize + hiddenSize);
b2 = theta ( 2 * hiddenSize * visibleSize + hiddenSize + 1:end );

% Cost and Gradient variables (your code needs to Compute These values). 
 % Here, WE initialize them to zeros.
cost = 0 ;
W1grad = zeros (size (W1));
W2grad = zeros (size (W2));
b1grad = zeros (size (b1));
b2grad = zeros (size (b2));

%% ---------- YOUR CODE HERE ----------------------------------- ---
% Instructions: Compute the cost / Optimization Objective J_sparse (W, b) for the Sparse Autoencoder,
 %                 and the CORRESPONDING gradients W1grad, W2grad, b1grad, b2grad.
 %
% W1grad, W2grad, b1grad and b2grad should be computed using Backpropagation.
 % Note that W1grad has the same Dimensions as W1, b1grad has the same Dimensions
 % as b1, etc. Your code should set W1grad to be the partial derivative of J_sparse (W , b) with 
% respect to W1. Ie, W1grad (i, J) should be the partial derivative of J_sparse (W, b) 
 % with respect to the input Parameter W1 (i, J). THUS, W1grad should be equal to the TERM 
 % [( 1 / m) \ Delta W ^ {( 1 )} + \ lambda W ^ {( 1 )}] in the Last Block  of pseudo-code in Section 2.2  
% of the Lecture Notes ( and similarly for W2grad , b1grad, b2grad).
 %
% Stated Differently, IF WE were using BATCH Gradient descent to optimize the Parameters,
 % the Gradient descent Update to W1 W1 would be: = W1 - alpha * W1grad, and similarly for W2, b1, b2. 
 % 

Jcost = 0 ;% direct error
Jweight = 0 ;% weight penalty
Jsparse = 0 ;% sparsity penalties
[n ,m] = size (Data);% m is the number of samples, n is the sample number of features

% forward algorithm to calculate the linear combination of neural network node values ??and active values
Z2 = W1 * Data + repmat (b1, 1 , m);% Note that vectors b1 sure to copy the column matrix expansion into m
A2 = sigmoid (Z2);
Z3 = W2 * A2 + repmat (b2, 1 , m);
A3 = sigmoid (Z3);

% calculated prediction error generated
Jcost = ( 0.5 / m) * sum (sum ((A3-Data).^ 2 ));

% penalty term weights are calculated
Jweight = ( 1 / 2 ) * (sum (sum (W1.^ 2 )) + sum (sum (W2.^ 2 )));

% calculated dilution Rules
rho = ( 1 / m).* sum (A2, 2 );% calculated average value of the first vector of hidden layer
Jsparse = sum (sparsityParam.* log (sparsityParam./ rho) +( 1 -sparsityParam).* log (( 1 -sparsityParam)./ ( 1 - rho)));

% total loss function expression
cost = Jcost + lambda * Jweight + beta * Jsparse;

% reverse algorithm derived error values ??for each node
d3 = - (Data-A3).* sigmoidInv (Z3);
Sterm = beta * (-sparsityParam./rho + ( 1 -sparsityParam)./ ( 1 -rho));% due to the additional sparse Rules, so
                                                              % when calculating the partial derivatives of the need to introduce
D2 = (W2' *d3 + repmat (Sterm, 1, m)).* sigmoidInv (Z2);

% calculated W1grad
W1grad = W1grad + D2 * Data' ; 
W1grad = ( 1 / m) * W1grad + lambda * W1;

% calculated W2grad  
W2grad = W2grad + d3 *  A2' ; 
W2grad = ( 1 / m).* W2grad + lambda * W2;

% calculated b1grad
b1grad = b1grad + sum (D2, 2 );
b1grad = ( 1 / m) * b1grad;% Note that the partial derivative of b is a vector, so here the value of each row should add up

% calculated b2grad
b2grad = b2grad + sum (d3, 2 );
b2grad = ( 1 / m) * b2grad;



%%% Second method, a sample of each treatment, slow
 % m = size (Data, 2 );
 % rho = zeros (size (b1));
 % for i = 1 : m
 %% Feedforward
 % A1 = Data (:, i);
 % Z2 = W1 * A1 + b1;
 % A2 = sigmoid (Z2);
 % Z3 = W2 * A2 + b2;
 % A3 = sigmoid (Z3);
 %% cost = cost + (A1-A3) ' * (A1-A3) * 0.5; 
% rho = rho + A2;
 % End 
% rho = rho / m;
 % Sterm = beta * (-sparsityParam./rho + ( 1 -sparsityParam). / ( 1 - rho));
 %% Sterm = beta * 2 * rho;
 % for i = 1 : m
 %% Feedforward
 % A1 = Data (:, i);
 % Z2 = W1 * A1 + b1;
 % A2 = sigmoid (Z2);
 % Z3 = W2 * A2 + b2;
 % A3 = sigmoid (Z3);
 % cost = cost + (A1-A3) ' * (A1-A3) * 0.5; 
%% Backpropagation
 % delta3 = (A3-A1). * A3. * ( 1 - A3) ;
 % delta2 = (W2 ' * delta3 + Sterm). * A2. * (1-A2); 
% W2grad = W2grad + delta3 * A2 ' ; 
% b2grad = b2grad + delta3;
 % W1grad = W1grad + delta2 * A1 ' ; 
% b1grad = b1grad + delta2;
 % End
% 
% KL = sparsityParam * log (sparsityParam. / Rho) + ( 1 -sparsityParam) * log (( 1 -sparsityParam). / ( 1 - rho));
 %% KL = rho. ^ 2 ;
 % cost = cost / m ;
 % cost = cost + sum (sum (W1. ^ 2 )) * lambda / 2.0 + sum (sum (W2. ^ 2 )) * lambda / 2.0 + beta * sum (KL);
 % W2grad = W2grad. / m + lambda * W2;
 % b2grad = b2grad. / m;
 % W1grad = W1grad. / m + lambda * W1;
 % b1grad = b1grad. / m;


% ------------------------------------------------- ------------------
% After Computing the cost and Gradient, WE will convert the gradients Back
 % to a Vector format (Suitable for minFunc). SPECIFICALLY, WE will unroll
 % your Gradient matrices into a Vector.

Grad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];

end

% ------------------------------------------------- ------------------
% Here ' s an implementation of the sigmoid function, which you may Find Useful 
% in your computation of the costs and the gradients. This Inputs a (Row or 
% column) Vector (say (Z1, Z2, Z3)) and Returns ( f (z1), f (z2), f (z3)).

function sigm = sigmoid (x)

    sigm = 1 ./ ( 1 + exp (- x));
 end

% sigmoid function of the inverse function
function sigmInv = sigmoidInv (x)

    sigmInv = sigmoid (x).* ( 1 - sigmoid (x));
 end