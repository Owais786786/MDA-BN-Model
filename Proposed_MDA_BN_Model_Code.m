% This code is written in Matlab R2020b version
clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 1: Load our trained MDA-BN model
load('TrainedModel.mat');
net = TrainedModel;
% analyzeNetwork(TrainedModel)
%% Step 2: Get User input for modality selection
opt = inputdlg( '0: Single Modality Data, 1: Heterogeneous Data' ,'Sample' , [1 50]);

if opt{1} == '0'
   [fn1,pn1] =  uigetfile('*.bmp','Select X-Ray or CT Image');
   img1 = imread(strcat(pn1,fn1));
   [predLabel1,probScore1] = classify(net,img1); 
   MultipleCAMs1 = getMultipleCAMs(net,img1);
   figure(1)
   subplot(2,6,1);imshow(img1,[]);title('Input Image');
   for i=1:5
   subplot(2,6,i+1); imshow(MultipleCAMs1{i});
   title(strcat('Activation:F',num2str(i)));
   end
   subplot(2,6,[7:12]);b = barh(probScore1,'FaceColor' , 'flat');
   b.CData(1,:) = [0.6350 0.0780 0.1840];
   b.CData(2,:) = [0.4660 0.6740 0.1880];
   xlim([0 1]);
   title('Confidence Score (C.S)');
   yticklabels({'COVID-19+','COVID-19-'});

elseif opt{1} == '1'
   [fn1,pn1] =  uigetfile('*.bmp','Select X-Ray Image');
   [fn2,pn2] =  uigetfile('*.bmp','Select CT Image');
   %%
   img1 = imread(strcat(pn1,fn1));
   [predLabel1,probScore1] = classify(net,img1);
   MultipleCAMs1 = getMultipleCAMs(net,img1);
   figure(1)
   subplot(4,6,1);imshow(img1,[]);title('Input Image');
   for i=1:5
   subplot(4,6,i+1); imshow(MultipleCAMs1{i});
   title(strcat('Activation:F',num2str(i)));
   end
   subplot(4,6,[7:12]);b = barh(probScore1,'FaceColor' , 'flat');
   b.CData(1,:) = [0.6350 0.0780 0.1840];
   b.CData(2,:) = [0.4660 0.6740 0.1880];
   xlim([0 1]);
   title('Confidence Score (C.S)');
   yticklabels({'COVID-19+','COVID-19-'});
   
   %%
   img2 = imread(strcat(pn2,fn2));
   [predLabel2,probScore2] = classify(net,img2);
   MultipleCAMs2 = getMultipleCAMs(net,img2);
   figure(1)
   subplot(4,6,13);imshow(img2,[]);title('Input Image');
   for i=1:5
   subplot(4,6,i+13); imshow(MultipleCAMs2{i});
   title(strcat('Activation:F',num2str(i)));
   end
   subplot(4,6,[19:24]);b = barh(probScore2,'FaceColor' , 'flat');
   b.CData(1,:) = [0.6350 0.0780 0.1840];
   b.CData(2,:) = [0.4660 0.6740 0.1880];
   xlim([0 1]);
   title('Confidence Score (C.S)');
   yticklabels({'COVID-19+','COVID-19-'});
else
    error('Invalid input option');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Required Functions for Generating Multiple CAM Images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MultipleCAMs = getMultipleCAMs(net,im)
layerNum = [130 125 120 115 110];
MultipleCAMs = [];
for j = 1 : length(layerNum)
    layerName =net.Layers(layerNum(j)).Name ; % MobileNet
    classActivationMap = activations(net,im,layerName);
    MultipleCAMs{j} = CAMshow(im,classActivationMap);
end
end
function combinedImage = CAMshow(im,CAM)

imSize = size(im);
CAM = imresize(CAM,imSize(1:2));
CAM = normalizeImage(CAM);
CAM(CAM<0.2) = 0;
cmap = jet(255).*linspace(0,1,255)';
CAM = ind2rgb(uint8(CAM*255),cmap)*255;
combinedImage = double((im))/2 + CAM;
combinedImage = normalizeImage(combinedImage)*255;
combinedImage = uint8(combinedImage);
end
function N = normalizeImage(I)
minimum = min(I(:));
maximum = max(I(:));
N = (I-minimum)/(maximum-minimum);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%