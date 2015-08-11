function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Computes cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

pred=X*theta;
h=sigmoid(pred);
m1=log(h);
m2=1.-h;
m2=log(m2);
j1=(-y)'*m1;
j2=(1.-y)'*m2;
j3= (lambda/2)*sum(theta([2:size(theta)]).^2);

J=(j1-j2+j3)/m;

grad=X'*(h-y);
grad=(1/m).*grad;
grad([2:size(theta)])=grad([2:size(theta)])+(lambda/m)*theta([2:size(theta)]);

% =============================================================

grad = grad(:);

end
