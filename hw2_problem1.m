close all;

% PROBLEM 1
%% Part a
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
testimages = loadMNISTImages('test-images.idx3-ubyte');
testlabels = loadMNISTLabels('test-labels.idx1-ubyte');

%% Part b
labels = labels';
labels(labels == 0) = 10;
labels = dummyvar(labels);  % one-hot encoding

testlabels = testlabels';
testlabels(testlabels == 0) = 10;
testlabels = dummyvar(testlabels);  % one-hot encoding

trainFcn = 'trainscg';

%% Part c
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;

%% Part d
net = patternnet();

labels = labels';

[net,tr] = train(net, images, labels);

y = net(images);
performance_netdata(10/10) = perform(net, labels, y);
tind = vec2ind(labels);
yind = vec2ind(y);
percentErrors_netdata(10/10) = sum(tind ~= yind)/numel(tind);

%% Part e
new_train_images = images(:,1:20000);
new_train_labels = labels(:,1:20000);
% these ratios change for the table in the pdf
% trainRatio: (20,000 - y) / 20,000
% valRatio: y / 20,000
%     Y -> Acc. Testing
%  3000 -> 91.05%
%  6000 -> 89.9%
%  9000 -> 88.53%
% 12000 -> 87.93%
% 15000 -> 89.26%

net.divideParam.trainRatio = 17000/20000;
net.divideParam.valRatio = 3000/20000;

net = patternnet();

[net,tr] = train(net, new_train_images, new_train_labels);

y = net(new_train_images);
performance_netdata(10/10) = perform(net, labels, y);
tind = vec2ind(labels);
yind = vec2ind(y);
percentErrors_netdata(10/10) = sum(tind ~= yind)/numel(tind);

%% Part f
% accuracy = (18210 / 20 000) * 100 = 91.05%
% y = 3000

%% Part g
confusionPlot = plotconfusion(testlabels, y);
saveas(confusionPlot, 'problem_1_g.png');
