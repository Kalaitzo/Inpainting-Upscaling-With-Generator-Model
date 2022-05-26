%% Task 2.3
clc;
clear;
close all;

%% Load the Generator
generator = load('data21.mat');
A1 = generator.A_1;
A2 = generator.A_2;
B1 = generator.B_1;
B2 = generator.B_2;

%% Load the Matricies
data = load("data23.mat");
X_ideal = data.X_i;  % The ideal vectors *NOT FOR PROCESSING*
X_processing = data.X_n;  % The vectors with added noise

%% Choose the data and initialize the transformation matrix
N = 49; 
T_size = [N, 784];

mean = 1/16;
T = zeros(T_size);
cnt1 = 0;
cnt2 = 0;
for i=0:48  % Algorithm to make the transformation matrix T
    if mod(i,7) == 0  % Change every seven lines to place to the correct cells
        cnt1 = 0;
        cnt2 = 4 * 28 * int16(i/7);
    end
    for j = 0*28 : 28 : 3*28
        T(i+1, (j+1)+cnt2+cnt1:(j+4)+cnt2+cnt1) = mean; % Set the quads
    end
    cnt1 = cnt1 + 4;
end

X_ideal = reshape(X_ideal,28,28,4);% The ideal vectors *NOT FOR PROCESSING*

Xn =  reshape(X_processing,N,1,4); % The 49 x 1 vectors for processing

Xn_paint = Xn;
Xn_paint = reshape(Xn_paint,7,7,4); % Enlarge these vectors to display them

%% Calculate Cost
inputs_size = [10, 1, 10];
costs_size = [1500,4,4];
generator_inputs = randn(inputs_size);  % Initialize the generator inputs
costs = zeros(costs_size);  % Initialize a cost matrix

lr = 0.005;  % Learning rate fro gradient descent

for i=1:4 % The amount Z's that we try as generator inputs
    Z = generator_inputs(:,:,i);  % Generator Input
    figure(i)
    for j=1:4
        Z_tmp = Z;  % Try the same Z for every target each time
        Xn_j = Xn(:,:,j);  % The generator Targets
        
        % Initialize Adam Parameters
        lambda = 1;
        power = 0;
        c = 10^-6;
        for itter=1:1500
            % First Layer
            W1 = A1 * Z_tmp + B1;  % Pass through the first layer
            Z1 = ReLU(W1);  % Pass through the Activation Function (ReLU)
            % Second Layer
            W2 = A2 * Z1 + B2;  % Pass through the second layer
            X = sigmoid(W2);  % Pass through the Activation Function (Sigmoid)

            J = cost(Z_tmp,Xn_j,X,N,T); % Calculate the cost
            costs(itter,j,i) = J;
            
            % Calculate the cost J(Z) gradient
            u2 = derivativePhi(Xn_j,X,T);
            v2 = u2.* DerivativeSigmoid(W2);
            u1 = A2' * v2;
            v1 = u1.* DerivativeReLU(W1);
            u0 = A1' * v1;
            grad = N * u0 + 2 * Z_tmp;

            % Adam normalization
            power = (1-lambda) * power + lambda * grad.^2;
            lambda = 0.001; % Change lambda to a small value

            % Gradient Descent Algorithm
            Z_new = Z_tmp - lr * grad./sqrt(power +  c);
            Z_tmp = Z_new;
        end
        subplot(3,4,j)
        imshow(X_ideal(:,:,j))
        subplot(3,4,j+4)
        imshow(kron(Xn_paint(:,:,j),ones(4)))
        subplot(3,4,j+8)
        imshow(reshape(X,28,28))
    end
end 
%% Plot the costs
for i=1:4    
    figure()
    plot(costs(:,:,i))
    ylabel('J(Z)')
    xlabel('Iteration')
    title('Costs')
    legend('Cost for 1st eight', ...
           'Cost for 2nd eight', ...
           'Cost for 3rd eight', ...
           'Cost for 4th eight')
end

%% Functions
function J = cost(Z,Xn,X,N,T) % Cost Function
    J = N * log(norm((Xn - T * X))^2) + norm(Z)^2;
end

function u2 = derivativePhi(Xn,X,T) % Phi Derivative
    u2 = (-2/(norm(Xn - T*X)^2)) * T' * (Xn - T*X);
end


function Z1 = ReLU(W)  % Activation ReLU
    Z1 = max(W,0);
end


function f1_der = DerivativeReLU(W)  % ReLU Derivative
    W(W(:,:)<=0) = 0;
    W(W(:,:)>0) = 1;
    f1_der = W;
end


function X = sigmoid(W)  % Activation Sigmoid
    X = 1./(1 + exp(W));
end


function f2_der = DerivativeSigmoid(W)  % Sigmoid Derivative
    f2_der = -exp(W)./((1 + exp(W)).^2);
end