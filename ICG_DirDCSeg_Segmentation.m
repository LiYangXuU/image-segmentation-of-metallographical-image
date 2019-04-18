function runtime = ICG_DirDCSeg_Segmentation(trial_name)
% ICG_DCSeg_Segmentation applies fast-UCM segmentation on specified image

%	****************************************************************
%	This code extends the Sketch Tokens code 
%	Joseph J. Lim, C. Lawrence Zitnick, and Piotr Dollar
%	"Sketch Tokens: A Learned Mid-level Representation for Contour 
%	and and Object Detection" CVPR2013
%   ****************************************************************
%
%   ****************************************************************


	if ~exist('trial_name','var'),trial_name = 'cvpr_2014';end
	
	image_dir = './3rdparty/BSR/BSDS500/data/images/test/';
	out_dir = ['./3rdparty/BSR/BSDS500/data/' trial_name '/test/'];
	
	addpath('Tools');
	
	matlabpool;
    
	% Create two directories, one for segments, one for edges
	mkdir(out_dir);
    mkdir([out_dir '/segments']);
    mkdir([out_dir '/edges']);
    
    % load the model
	model = [];
    load('./models/CVPR_2014_PRE_TRAINED_MODEL');
    
	% Get all jpg and png files in input directory
    filenames = ICG_ListFilename(image_dir,'*.jpg;*.png');
    
    % Iterate through images (parallel processing)
    runtime = zeros(1,numel(filenames));
    parfor img_id = 1 : numel(filenames) 
        % Read image
        img = imread([image_dir filenames{img_id}]);
        
        % Apply Segmenter
		[ucm2,ucm,grads,runtime(img_id)] = ICG_DCSeg_Segmentation(img,model);
        
		% Write outputs to corresponding files
        imwrite(ucm,[out_dir 'edges/' filenames{img_id}(1:end-4) '.png']);
        iSaveX([out_dir 'segments/' filenames{img_id}(1:end-4) '.mat'],ucm2);
        
    end
    matlabpool close;
    avg_time = mean(runtime);
    disp(['Segmented ' num2str(numel(filenames)) ' images requiring on average ' ...
        num2str(avg_time) ' seconds per image!'])
    
end 

% helper function for saving in parfor
function iSaveX(fname,ucm2)
    save(fname,'ucm2');
end
