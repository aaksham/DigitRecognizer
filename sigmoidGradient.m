function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, should return
%   the gradient for each element.

g = zeros(size(z));
g1=sigmoid(z);
g2=1.-g1;
g=g1.*g2;
% =============================================================
end