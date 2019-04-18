function segmentation


    addpath('Tools');

    img_name = 'set4_013.tiff';
	
	% load the image 
	I = imread(img_name);
	img = graytrgb(I);
	load('./models/TRAINED_MODEL');
	
    tic;
    [ucm1,ucm] = ICG_DCSeg_Segmentation(img, model);
    
    k = [0.08, 0.1, 0.15];
    % threshold
	labels2 = bwlabel(ucm1 <= k(1));
	labels_1 = labels2(2:2:end, 2:2:end);
    labels2 = bwlabel(ucm1 <= k(2));
	labels_2 = labels2(2:2:end, 2:2:end);
    labels2 = bwlabel(ucm1 <= k(3));
	labels_3 = labels2(2:2:end, 2:2:end);

    close all,figure;
	subplot(221);imshow(img);
    subplot(222);imshow(ICG_LabelToMeanImage(labels_1,img));
    subplot(223);imshow(ICG_LabelToMeanImage(labels_2,img));
    subplot(224);imshow(ICG_LabelToMeanImage(labels_3,img));
    ICG_ToolMaximizeFigure;
	toc;
