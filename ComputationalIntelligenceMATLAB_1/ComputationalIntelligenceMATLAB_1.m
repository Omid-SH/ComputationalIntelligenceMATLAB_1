%% Problem 1
clear; clc
load('Ex1.mat')
in_speed = speed;
in_no = NOemission;
out_fuel = fuelrate;
clear speed fuelrate NOemission

%% A
scatter3(in_no, in_speed, out_fuel, 2)
xlabel('NO emission')
ylabel('Speed')
zlabel('Fuel rate')

%% B
X = [in_no(1:end); in_speed(1:end)];
Y = out_fuel(1:end);

X_train = [in_no(1:700)', in_speed(1:700)'];
Y_train = out_fuel(1:700)';

X_valid = [in_no(701:end)', in_speed(701:end)'];
Y_valid = out_fuel(701:end)';

%% C
Linear_md = fitrlinear(X_train, Y_train)
fprintf('Accuracy on train data is %i\n',loss(Linear_md, X_train, Y_train));
fprintf('Accuracy on validation data is %i\n',loss(Linear_md, X_valid, Y_valid));

%%

fitglm(X_train, Y_train)

%% D
Z = max(Y_train);
Z_train = log((Z-Y_train)./Y_train);
Linear_md = fitrlinear(X_train, Z_train);
logestic_valid_out = sum(Linear_md.Beta .* X_valid') + Linear_md.Bias;
logestic_valid_out = Z ./ (1 + exp(logestic_valid_out));
fprintf('Accuracy on validation data is %i\n',immse(logestic_valid_out, Y_valid'));

%% E
net = fitnet(100);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:700;
net.divideParam.valInd   = 701:length(Y);
net = train(net, X, Y)

%% Problem 2
clear; clc
load('Ex2.mat')

scatter3(TrainData(1,:), TrainData(2,:), TrainData(3,:), 10, TrainData(4,:)); % draw 2 classes of data

%% A

%% feedforwardnet with 1 layer small hiddenLayerSize
net = feedforwardnet(2);
view(net)

%% feedforwardnet with 1 layer big hiddenLayerSize
net = feedforwardnet(25);
view(net)

%% feedforwardnet with 2 layer
net = feedforwardnet([10, 10]);
view(net)

%% patternnet with 1 layer small hiddenLayerSize
net = patternnet(2);
view(net)

%% patternnet with 1 layer big hiddenLayerSize
net = patternnet(25);
view(net)

%% patternnet with 2 layer
net = patternnet([10, 10]);
view(net)

%% Setup and train network for part A
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100; 
        
% Train the Network for Part B
[net,tr] = train(net,TrainData(1:3,:),TrainData(4,:));

%% Calculate part A network accuracy on validation part of the main train data
val_data = TrainData(:,tr.valInd);

ans_net_in_val_data = net(val_data(1:3,:));
val_size = size(val_data,2);
number_of_errors = sum(abs((round(ans_net_in_val_data) > 0) - val_data(4,:)));
accuracy_in_val_data = (val_size - number_of_errors) / val_size * 100;

fprintf('Accuracy on validation part of the main train data is %g %%\n',accuracy_in_val_data);

%% Calculate part A network accuracy on whole part of the main train data
ans_net_in_train_data = net(TrainData(1:3,:));
train_size = size(TrainData,2);
number_of_errors = sum(abs((round(ans_net_in_train_data) > 0) - TrainData(4,:)));
accuracy_in_whole_data = (train_size - number_of_errors) / train_size * 100;

fprintf('Accuracy on whole of the main train data is %g %%\n',accuracy_in_whole_data);

%% Calculate part A output for the main test data
ans_net_in_test_data = net(TestData);
testlabel_a = abs((round(ans_net_in_test_data) > 0));
save testlabel_a

%% B (prepare data for this part)
Train_out(1,:) = (TrainData(4,:)==0); % generate first output
Train_out(2,:) = (TrainData(4,:)==1); % generate second output
Train_out = double(Train_out); % ouput size is a 2*N 
Train_in = TrainData(1:3,:); % input size is a 3*N  

%% feedforwardnet with 1 layer small hiddenLayerSize
net = feedforwardnet(2);
view(net)

%% feedforwardnet with 1 layer big hiddenLayerSize
net = feedforwardnet(25);
view(net)

%% feedforwardnet with 2 layer
net = feedforwardnet([10, 10]);
view(net)

%% patternnet with 1 layer small hiddenLayerSize
net = patternnet(2);
view(net)

%% patternnet with 1 layer big hiddenLayerSize
net = patternnet(25);
view(net)

%% patternnet with 2 layer
net = patternnet([10, 10]);
view(net)

%% Setup and train network for part B
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100; 

% Train the Network for Part B
[net,tr] = train(net,Train_in,Train_out);

%% Calculate part B network accuracy on validation part of the main train data
val_data = TrainData(:,tr.valInd);

ans_net_in_val_data = net(val_data(1:3,:));
val_size = size(val_data,2);
number_of_errors = sum(abs(double(ans_net_in_val_data(2,:)>ans_net_in_val_data(1,:)) - val_data(4,:)));
accuracy_in_val_data = (val_size - number_of_errors) / val_size * 100;

fprintf('Accuracy on validation part of the main train data is %g %%\n',accuracy_in_val_data);

%% Calculate part B network accuracy on whole part of the main train data
ans_net_in_train_data = net(TrainData(1:3,:));
train_size = size(TrainData,2);
number_of_errors = sum(abs(double(ans_net_in_train_data(2,:)>ans_net_in_train_data(1,:)) - TrainData(4,:)));
accuracy_in_whole_data = (train_size - number_of_errors) / train_size * 100;

fprintf('Accuracy on whole part of the main train data is %g %%\n',accuracy_in_whole_data);

%% Calculate part B output for the main test data
ans_net_in_test_data = net(TestData);
testlabel_b = abs(double(ans_net_in_test_data(2,:)>ans_net_in_test_data(1,:)));
save testlabel_b
