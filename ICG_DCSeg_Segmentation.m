function [ucm2,ucm,gradient_orientations,times] = ICG_DCSeg_Segmentation(img, model)
% ICG_DCSeg_Segmentation applies fast-UCM segmentation on specified image

%	****************************************************************
%	This code extends the Sketch Tokens code 
%	Joseph J. Lim, C. Lawrence Zitnick, and Piotr Dollar
%	"Sketch Tokens: A Learned Mid-level Representation for Contour 
%	and and Object Detection" CVPR2013
%   ****************************************************************
%
%   ****************************************************************

    addpath('./3rdparty/BSR/grouping/lib');
    addpath('./3rdparty/SketchTokens-master');
    addpath('Tools');
    
	 % First step: predict gradient orientation signals 
    t = cputime; 
    [gradient_orientations,max_strength] = ICG_OrientedGradientSignalPrediction( img, model ); 
    
    gradient_orientations = gradient_orientations ./ max(gradient_orientations(:));
	
	% Apply oriented watershed based grouping
    [ucm,ucm2] = ICG_Contours2ucm(gradient_orientations, 'both');
    ucm = ucm .* (max(max_strength(:)) / max(ucm(:)));
    ucm2 = ucm2 .* (max(max_strength(:)) / max(ucm2(:)));
    times = cputime-t;
	
	    