function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X=[ones(m,1) X];
pred1=sigmoid(X*Theta1');
%size(X)
%size(Theta1)
pred1=[ones(m,1) pred1];
%size(pred1)
%size(Theta2)
pred2=sigmoid(pred1*Theta2');
[x,ix]=max(pred2,[],2);
p=ix;



% =========================================================================


end
