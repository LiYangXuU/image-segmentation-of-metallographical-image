function [codebook_orientation_image,the_magnitudes,S] = ICG_OrientedGradientSignalPrediction( I, model, stride )
% ICG_DCSeg_Segmentation applies fast-UCM segmentation on specified image

%	****************************************************************
%	This code extends the Sketch Tokens code 
%	Joseph J. Lim, C. Lawrence Zitnick, and Piotr Dollar
%	"Sketch Tokens: A Learned Mid-level Representation for Contour 
%	and and Object Detection" CVPR2013
%   ****************************************************************
%
%   ****************************************************************


    if nargin<3
        stride=2;
    end
    
    % we do need the dollar toolbox
    addpath('./3rdparty/toolbox/channels');
    addpath('./3rdparty/SketchTokens-master');	
    % compute features
    sizeOrig=size(I);
    opts=model.opts;
    opts.inputColorChannel = 'rgb';
    I = imPad(I,opts.radius,'symmetric');
    chns = stChns( I, opts );
    [cids1,cids2] = computeCids(size(chns),opts);
    chnsSs = convBox(chns,opts.cellRad);

    % run forest on image
    S = stDetectMex( chns, chnsSs, model.thrs, model.fids, model.child, ...
      model.distr, cids1, cids2, stride, opts.radius, opts.nChnFtrs );

    % get probability maps
    S = permute(S,[2 3 1]) * (1/opts.nTrees);
    S = imResample( S, stride );
    cr=size(S); cr=cr(1:2)-sizeOrig(1:2);
    if any(cr)
        S=S(1:end-cr(1),1:end-cr(2),:);
    end
    S = max(min(1,S),0);
    
    % Predict the local gradient orientation
    the_grads = squeeze(mean(model.codebook_gradients,1));
    the_grads(end,:) = [0 0 0 0 0 0 0 0];
    codebook_orientation_image = zeros(size(S,1)*size(S,2),8);
    
    S_reshaped = reshape(S,size(S,1)*size(S,2),size(S,3));
    P = S(:,:,1:end-1);
    the_magnitudes = sum(P,3);
    the_magnitudes_reshaped = reshape(the_magnitudes,size(the_magnitudes,1) ...
        *size(the_magnitudes,2),size(the_magnitudes,3));
    
    % Predict all orientations    
    for id = 1 : size(the_grads,2)
        codebook_orientation_image(:,id) = ...
            (S_reshaped*the_grads(:,id)) .*the_magnitudes_reshaped;
    end
    codebook_orientation_image = reshape(codebook_orientation_image,[size(S,1),size(S,2),8])./255;
    codebook_orientation_image = max(min(1,codebook_orientation_image),0);
        


end

function [cids1,cids2] = computeCids( siz, opts )
    % construct cids lookup for standard features
    radius=opts.radius;
    s=opts.patchSiz;
    nChns=opts.nChns;
    
    ht=siz(1);
    wd=siz(2);
    assert(siz(3)==nChns);
    
    nChnFtrs=s*s*nChns;
    fids=uint32(0:nChnFtrs-1);
    rs=mod(fids,s);
    fids=(fids-rs)/s;
    cs=mod(fids,s);
    ch=(fids-cs)/s;
    cids = rs + cs*ht + ch*ht*wd;
    
    % construct cids1/cids2 lookup for self-similarity features
    n=opts.nCells;
    m=opts.cellStep;
    nCellTotal=(n*n)*(n*n-1)/2;
    
    assert(mod(n,2)==1); n1=(n-1)/2;
    nSimFtrs=nCellTotal*nChns;
    fids=uint32(0:nSimFtrs-1);
    ind=mod(fids,nCellTotal);
    ch=(fids-ind)/nCellTotal;
    
    k=0;
    for i=1:n*n-1,
        k1=n*n-i;
        ind1(k+1:k+k1)=(0:k1-1);
        k=k+k1;
    end
    k=0;
    for i=1:n*n-1,
        k1=n*n-i;
        ind2(k+1:k+k1)=(0:k1-1)+i;
        k=k+k1;
    end
    
    ind1=ind1(ind+1);
    rs1=mod(ind1,n);
    cs1=(ind1-rs1)/n;
    ind2=ind2(ind+1);
    rs2=mod(ind2,n);
    cs2=(ind2-rs2)/n;
    
    rs1=uint32((rs1-n1)*m+radius);
    cs1=uint32((cs1-n1)*m+radius);
    rs2=uint32((rs2-n1)*m+radius);
    cs2=uint32((cs2-n1)*m+radius);
    
    cids1 = rs1 + cs1*ht + ch*ht*wd;
    cids2 = rs2 + cs2*ht + ch*ht*wd;
    
    % combine cids for standard and self-similarity features
    cids1=[cids cids1];
    cids2=[zeros(1,nChnFtrs) cids2];
end

function ftrs = stComputeSimFtrs( chns, opts )
% Compute self-similarity features.
    n=opts.nCells;
    if(n==0),
        ftrs=[];
        return;
    end
    nSimFtrs=opts.nSimFtrs;
    nChns=opts.nChns;
    m=size(chns,4);
    
    inds = ((1:n)-(n+1)/2)*opts.cellStep+opts.radius+1;
    chns=reshape(chns,opts.patchSiz,opts.patchSiz,nChns*m);
    chns=convBox(chns,opts.cellRad);
    chns=reshape(chns(inds,inds,:,:),n*n,nChns,m);
    ftrs=zeros(nSimFtrs/nChns,nChns,m,'single');
    k=0;
    for i=1:n*n-1
        k1=n*n-i;
        ftrs(k+1:k+k1,:,:)=chns(1:end-i,:,:)-chns(i+1:end,:,:);
        k=k+k1;
    end
    ftrs = reshape(ftrs,nSimFtrs,m)';
    % % For m=1, the above should be identical to the following:
    % [cids1,cids2]=computeCids(size(chns),opts); % see stDetect.m
    % chns=convBox(chns,opts.cellRad); k=opts.nChnFtrs;
    % cids1=cids1(k+1:end)-k+1; cids2=cids2(k+1:end)-k+1;
    % ftrs=chns(cids1)-chns(cids2);
end
