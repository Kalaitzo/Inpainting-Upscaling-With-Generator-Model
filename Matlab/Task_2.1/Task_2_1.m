%% Task 2.1
clc;
clear;
close all;
%% Import the data of the generator
generator = load('data21.mat');
A1 = generator.A_1;
A2 = generator.A_2;
B1 = generator.B_1;
B2 = generator.B_2;

%% Create inputs Z ~ N(0,1)
sz = [10, 1, 100];
inputs = randn(sz);  % Create the generator inputs

%% Calculate the Generator Output
output = zeros(784,1,100);

for i=1:100
    Z = inputs(:,:,i); % Take a signle input
    
    % First Layer
    W1 = A1 * Z + B1;  % Pass through the first layer
    Z1 = ReLU(W1);  % Pass through the Activation Function (ReLU)
    % Second Layer
    W2 = A2 * Z1 + B2;  % Pass through the second layer
    X = sigmoid(W2);  % Pass through the Activation Function (Sigmoid)

    output(:,:,i) = X; % Keep all the output vectors
end

%% Reshape the output to a picture
canvas = []; % Initialize the 10x10 picture "canvas"
row = []; % Initialize a temp row matix for the "canvas"
cnt = 0; % Initialize a counter so we go to the next row every 10 pictures

for i=1:100
    X_2D = reshape(output(:,:,i), 28, 28); % Make the output a 28x28 matrix
    row = [row, X_2D]; % Add that matrix to the current row
    cnt = cnt + 1; % Increase the counter
    if cnt == 10
        canvas = [canvas; row]; % Add the curent row to the canvas
        row = []; % Empty the row for the next pictures
        cnt = 0; % Set the counter to 0
    end
end

%% Show the image
imshow(canvas)  % Display the 100 eights

%% Activation Functions
function Z1 = ReLU(W) % Activation ReLU
    Z1 = max(W,0);
end

function X = sigmoid(W) % Activation Sigmoid
    X = 1./(1 + exp(W));
end